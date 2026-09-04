"""
Phase 3: OCR result -> field_classifier -> rule_engine -> one compliance report.
Phase 4 formalized the return type into the documented ComplianceReport schema.

The only file in backend/compliance/ that imports backend.ocr.schemas - a deliberate,
narrow exception to the package's self-containment, because run_full_compliance_pipeline's
whole job is to accept an already-completed OCRResult. backend/ocr/schemas.py itself has
no FastAPI dependency (pure pydantic), so this stays decoupled from the FastAPI app;
nothing here imports backend.api.* or backend.main.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from backend.compliance.field_classifier import classify_fields
from backend.compliance.llm_classifier import MODEL as LLM_MODEL
from backend.compliance.llm_classifier import LLMClassificationError, classify_fields_with_llm
from backend.compliance.report_schema import (
    FIELD_LABELS,
    ComplianceReport,
    ProductIdentification,
    build_compliance_report,
)
from backend.compliance.rule_engine import ClassifiedFields, Ruleset, load_ruleset, run_compliance_check
from backend.ocr.schemas import OCRResult

logger = logging.getLogger("legallense.compliance.pipeline")

VALID_CLASSIFIER_MODES = ("regex", "llm", "hybrid")


def _classifier_mode() -> str:
    """
    COMPLIANCE_CLASSIFIER_MODE controls which field classifier run_full_compliance_pipeline
    uses. Defaults to "regex" - unset, empty, or unrecognized values all mean "nothing
    changes from existing behavior unless explicitly opted in".
    """
    mode = os.getenv("COMPLIANCE_CLASSIFIER_MODE", "regex").strip().lower()
    if mode not in VALID_CLASSIFIER_MODES:
        if mode:
            logger.warning("Unknown COMPLIANCE_CLASSIFIER_MODE=%r, defaulting to 'regex'", mode)
        return "regex"
    return mode


def _enrich_fields(report_dict: Dict[str, Any], classified: ClassifiedFields) -> None:
    """Attach each field's classification confidence + matched detections (bbox included,
    for box-coloring) before handing off to build_compliance_report."""
    for field_dict in report_dict["fields"]:
        classified_field = classified.get(field_dict["field"])
        field_dict["confidence"] = classified_field.confidence if classified_field else None
        field_dict["matched_detections"] = (
            [d.model_dump() for d in classified_field.matched_detections] if classified_field else []
        )


def run_full_compliance_pipeline(
    ocr_result: OCRResult,
    profile: str = "e_commerce_product_listing",
    ruleset: Optional[Ruleset] = None,
    product: Optional[ProductIdentification] = None,
) -> ComplianceReport:
    """
    Run one already-completed OCRResult through field classification and the Rule 6
    compliance engine, returning a single ComplianceReport (see report_schema.py).

    `product` carries whatever product-identification context the caller already has
    (e.g. a persisted scan's productName/category/manufacturer); when omitted, only the
    OCR image filename is known and used.

    Classifier source is controlled by the COMPLIANCE_CLASSIFIER_MODE env var:
      - "regex" (default): field_classifier.classify_fields() only. Unchanged behavior.
      - "llm": llm_classifier.classify_fields_with_llm() (Groq). Falls back to the regex
        classifier automatically - with a logged warning and a note on the report - if the
        LLM is unavailable (no GROQ_API_KEY) or the API call fails for any reason (timeout,
        rate limit, malformed response), so a Groq outage never breaks a compliance check.
      - "hybrid": runs regex (always - free, instant, no network) and additionally tries
        the LLM once. The report is still built from the regex classification - the
        stable, existing default - but any field where the two disagree on the final
        verdict gets an explicit note flagging it for human review. This is a genuinely
        additive cross-check, not an automatic override of regex by the LLM.
    """
    ruleset = ruleset or load_ruleset()
    product = product or ProductIdentification()
    if product.image is None:
        product.image = ocr_result.image

    if not ocr_result.success:
        return build_compliance_report({
            "profile": profile,
            "ruleset_id": ruleset.ruleset_id,
            "ruleset_version": ruleset.version,
            "evaluated_rules": [],
            "fields": [],
            "violations": [],
            "exemptions": None,
            "overall_status": "Unknown",
            "notes": [
                "OCR did not succeed for this image"
                + (f": {ocr_result.error.message}" if ocr_result.error else "")
                + " - compliance cannot be evaluated.",
            ],
        }, product=product)

    mode = _classifier_mode()
    extra_notes: List[str] = []

    regex_classified: Optional[ClassifiedFields] = None
    llm_classified: Optional[ClassifiedFields] = None

    if mode in ("regex", "hybrid"):
        regex_classified = classify_fields(ocr_result.full_text, ocr_result.detections, ruleset=ruleset)

    if mode in ("llm", "hybrid"):
        try:
            llm_classified, usage, _expiry = classify_fields_with_llm(ocr_result.full_text, ocr_result.detections)
            extra_notes.append(
                f"LLM classification (Groq {LLM_MODEL}) used ~{usage.total_tokens} tokens "
                f"(~${usage.estimated_cost_usd:.6f})."
            )
        except LLMClassificationError as exc:
            logger.warning("LLM classification unavailable (%s); falling back to regex classifier", exc)
            extra_notes.append(f"LLM classification was unavailable ({exc}); used the regex classifier instead.")
            if mode == "llm" and regex_classified is None:
                regex_classified = classify_fields(ocr_result.full_text, ocr_result.detections, ruleset=ruleset)

    classified = llm_classified if mode == "llm" and llm_classified is not None else regex_classified

    report_dict = run_compliance_check(
        ocr_result.full_text, profile=profile, ruleset=ruleset, classified_fields=classified,
    )

    if mode == "hybrid" and llm_classified is not None and regex_classified is not None:
        llm_report_dict = run_compliance_check(
            ocr_result.full_text, profile=profile, ruleset=ruleset, classified_fields=llm_classified,
        )
        llm_by_field = {f["field"]: f for f in llm_report_dict["fields"]}
        for regex_field in report_dict["fields"]:
            llm_field = llm_by_field.get(regex_field["field"])
            if llm_field and llm_field["compliant"] != regex_field["compliant"]:
                label = FIELD_LABELS.get(regex_field["field"], regex_field["field"])
                extra_notes.append(
                    f"Classifier disagreement on {label}: regex says {regex_field['compliant']!r} "
                    f"(value={regex_field['extracted_value']!r}), LLM says {llm_field['compliant']!r} "
                    f"(value={llm_field['extracted_value']!r}) - worth a human look."
                )

    report_dict.setdefault("notes", [])
    report_dict["notes"].extend(extra_notes)

    _enrich_fields(report_dict, classified)

    return build_compliance_report(report_dict, product=product)
