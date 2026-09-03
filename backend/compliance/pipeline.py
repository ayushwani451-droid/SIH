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

from typing import Optional

from backend.compliance.field_classifier import classify_fields
from backend.compliance.report_schema import ComplianceReport, ProductIdentification, build_compliance_report
from backend.compliance.rule_engine import Ruleset, load_ruleset, run_compliance_check
from backend.ocr.schemas import OCRResult


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

    classified = classify_fields(ocr_result.full_text, ocr_result.detections, ruleset=ruleset)
    report_dict = run_compliance_check(
        ocr_result.full_text, profile=profile, ruleset=ruleset, classified_fields=classified,
    )

    # Enrich each field with its classification confidence (Phase 2 metadata) before
    # handing off to build_compliance_report - rule_engine's own FieldResult type stays
    # untouched; this merge is pipeline.py's job as the integration layer.
    for field_dict in report_dict["fields"]:
        classified_field = classified.get(field_dict["field"])
        field_dict["confidence"] = classified_field.confidence if classified_field else None

    return build_compliance_report(report_dict, product=product)
