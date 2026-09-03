"""
Phase 4: the stable, documented JSON schema for a compliance report.

This is the one shape GET /api/ocr/compliance/{scan_id} returns, and the shape a future
frontend should consume. Internal modules (rule_engine.run_compliance_check,
pipeline.run_full_compliance_pipeline) can keep evolving their internal working dict
shape, but should always end up building a ComplianceReport via build_compliance_report()
before crossing the API boundary - that function is the one place the internal dict shape
and this schema meet.

No UI is built against this in this phase - just the schema, documented, so a frontend
task can consume it later without re-deriving the shape from the API by trial and error.

Field naming is snake_case throughout, matching every other JSON API in this backend
(OCRResult, FieldResult, etc.) - NOT the camelCase used by the existing auth/manufacturer
endpoints (e.g. fullName, companyName). Whoever builds the frontend consumer should decide
once whether to camelCase at the API boundary or in the frontend client; not decided here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

FIELD_LABELS: Dict[str, str] = {
    "manufacturer_packer_importer_details": "Manufacturer / Packer / Importer Details",
    "common_generic_name_of_commodity": "Common / Generic Name of Commodity",
    "net_quantity": "Net Quantity",
    "month_and_year_of_manufacture_or_packing": "Month & Year of Manufacture / Packing",
    "maximum_retail_price_mrp": "Maximum Retail Price (MRP)",
    "unit_sale_price": "Unit Sale Price",
    "consumer_care_details": "Consumer Care Details",
    "country_of_origin": "Country of Origin",
}


class ProductIdentification(BaseModel):
    """
    Who/what this report is about ("Section 4: product identification"). Every field is
    optional because a report can be generated straight from a bare OCRResult with no
    persisted scan/product context at all (pipeline.run_full_compliance_pipeline's
    standalone usage) - populate whatever is actually known.
    """

    scan_id: Optional[str] = Field(None, description="Persisted scan id (e.g. 'SCN-...'); None if not from a stored scan")
    image: Optional[str] = Field(None, description="Source image filename")
    product_name: Optional[str] = None
    category: Optional[str] = None
    manufacturer: Optional[str] = None


class MatchedDetection(BaseModel):
    """
    One OCR detection field_classifier.py attributed to a field - just enough for a
    frontend to draw/color a bounding box, not the full Detection shape from
    backend/ocr/schemas.py (this module stays independent of that schema).
    """

    text: str
    bbox: List[int] = Field(default_factory=list, description="[x1, y1, x2, y2] in the original image's pixel coordinates")
    confidence: float = 1.0


class FieldVerdict(BaseModel):
    """One row of the field-wise verdict table - one per Rule 6 mandatory declaration."""

    field: str = Field(..., description="Machine key, e.g. 'maximum_retail_price_mrp'")
    field_label: str = Field(..., description="Human-readable label for display, e.g. 'Maximum Retail Price (MRP)'")
    status: Literal["present", "absent", "ambiguous"]
    verdict: Literal["Pass", "Fail", "Needs Review", "Not Applicable"] = Field(
        ..., description="Frontend-friendly derived verdict - see `compliant`/`conditional_field` for the raw logic behind it"
    )
    extracted_value: Optional[str] = Field(None, description="The OCR text this verdict was based on, if any")
    compliant: Union[bool, Literal["needs_review"]] = Field(
        ..., description="Raw engine verdict: True (pass), False (hard fail), or 'needs_review'"
    )
    conditional_field: bool = Field(
        False,
        description="True for fields (currently just country_of_origin) that are only mandatory in some "
                     "cases (e.g. imported products); their absence alone does not count as a violation",
    )
    reason: Optional[str] = Field(None, description="Plain-language explanation, present whenever verdict != 'Pass'")
    confidence: Optional[Literal["high", "medium", "low"]] = Field(
        None,
        description="How confidently field_classifier.py located this field in the OCR text "
                     "(Phase 2). None if classification wasn't used, or nothing was found at all.",
    )
    matched_detections: List[MatchedDetection] = Field(
        default_factory=list,
        description="Which raw OCR detection(s) (text + bbox) this verdict was based on, for a "
                     "frontend to draw/color-code bounding boxes against. Empty if classification "
                     "wasn't used or nothing was found for this field.",
    )


class Violation(BaseModel):
    """One entry in the violation list - every FieldVerdict whose verdict isn't 'Pass' or 'Not Applicable'."""

    field: str
    field_label: str
    severity: Literal["violation", "needs_review"] = Field(
        ..., description="'violation' = hard fail (compliant is False); 'needs_review' = ambiguous, not auto-failed"
    )
    reason: str


class ComplianceReport(BaseModel):
    """
    The stable compliance report returned by GET /api/ocr/compliance/{scan_id} and by
    pipeline.run_full_compliance_pipeline(). Treat this as an API contract: additive
    changes (new optional fields) are fine; renaming or removing fields is a breaking
    change and should bump schema_version.

    - product: who/what this report is about
    - fields: the field-wise verdict table, one row per Rule 6 declaration evaluated
    - violations: the subset of `fields` that isn't a clean Pass, with plain-language reasons
    - overall_status: the single top-line verdict a frontend would show first
    - overall_confidence: best-effort aggregate (the weakest link) of the per-field confidence indicators
    """

    schema_version: str = "1.0"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    product: ProductIdentification
    profile: str
    ruleset_id: str
    ruleset_version: str
    evaluated_rules: List[str] = Field(default_factory=list)

    fields: List[FieldVerdict] = Field(default_factory=list)
    violations: List[Violation] = Field(default_factory=list)
    exemptions: Optional[Dict[str, Any]] = None

    overall_status: Literal["Compliant", "Partially Compliant", "Non-Compliant", "Unknown"]
    overall_confidence: Optional[Literal["high", "medium", "low"]] = Field(
        None,
        description="Weakest confidence among fields that had a classification confidence at all; "
                     "None if classification wasn't used for this report",
    )

    notes: List[str] = Field(default_factory=list)


def _verdict_for(
    compliant: Union[bool, str], conditional_field: bool, status: str
) -> Literal["Pass", "Fail", "Needs Review", "Not Applicable"]:
    if conditional_field and status == "absent":
        return "Not Applicable"
    if compliant is True:
        return "Pass"
    if compliant == "needs_review":
        return "Needs Review"
    return "Fail"


_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def build_compliance_report(
    pipeline_result: Dict[str, Any],
    product: Optional[ProductIdentification] = None,
) -> ComplianceReport:
    """
    Build the stable ComplianceReport from the internal working dict shape produced by
    rule_engine.run_compliance_check() (optionally enriched with a "confidence" key per
    field by pipeline.py when classification was used). This is the one function that
    knows about both shapes.
    """
    product = product or ProductIdentification(image=(pipeline_result.get("source") or {}).get("image"))

    field_verdicts: List[FieldVerdict] = []
    for raw in pipeline_result.get("fields", []):
        field_key = raw["field"]
        field_verdicts.append(FieldVerdict(
            field=field_key,
            field_label=FIELD_LABELS.get(field_key, field_key.replace("_", " ").title()),
            status=raw["status"],
            verdict=_verdict_for(raw["compliant"], raw.get("conditional_field", False), raw["status"]),
            extracted_value=raw.get("extracted_value"),
            compliant=raw["compliant"],
            conditional_field=raw.get("conditional_field", False),
            reason=raw.get("reason"),
            confidence=raw.get("confidence"),
            matched_detections=[
                MatchedDetection(text=d.get("text", ""), bbox=d.get("bbox", []), confidence=d.get("confidence", 1.0))
                for d in raw.get("matched_detections", [])
            ],
        ))

    violations: List[Violation] = [
        Violation(
            field=fv.field, field_label=fv.field_label,
            severity="violation" if fv.verdict == "Fail" else "needs_review",
            reason=fv.reason or "",
        )
        for fv in field_verdicts
        if fv.verdict in ("Fail", "Needs Review")
    ]

    confidences = [fv.confidence for fv in field_verdicts if fv.confidence]
    overall_confidence = min(confidences, key=lambda c: _CONFIDENCE_ORDER[c]) if confidences else None

    return ComplianceReport(
        product=product,
        profile=pipeline_result["profile"],
        ruleset_id=pipeline_result["ruleset_id"],
        ruleset_version=pipeline_result["ruleset_version"],
        evaluated_rules=pipeline_result.get("evaluated_rules", []),
        fields=field_verdicts,
        violations=violations,
        exemptions=pipeline_result.get("exemptions"),
        overall_status=pipeline_result["overall_status"],
        overall_confidence=overall_confidence,
        notes=pipeline_result.get("notes", []),
    )
