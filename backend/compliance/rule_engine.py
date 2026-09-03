"""
Standalone Legal Metrology Rule 6 compliance engine (Phase 1).

Loads backend/data/legal_metrology_rules.json into typed models and runs
heuristic, regex-based checks for each of the 8 mandatory declarations under
Rule 6 of the Legal Metrology (Packaged Commodities) Rules, 2011, against a
block of OCR text.

Deliberately has NO dependency on the FastAPI app (main.py, api/ocr.py,
api/auth.py, etc.) so it can be imported, unit tested, and iterated on in
isolation. Wiring this into /api/ocr is a separate, later phase.

Several checks here are honest, best-effort heuristics rather than reliable
field detection (most notably common_generic_name_of_commodity, which we
cannot yet isolate from raw text without product-category matching). Where
that's true it's called out in the function's FieldResult.reason and in
comments, and the result is surfaced as "needs_review" rather than a false
pass or fail.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

DEFAULT_RULESET_PATH = Path(__file__).resolve().parents[1] / "data" / "legal_metrology_rules.json"

# Same regex as legal_metrology_rules.json's LMPC-R6-MANDATORY-DECLARATIONS.validations.mrp_format,
# kept as a fallback in case that rule/validation is ever missing from the loaded file.
FALLBACK_MRP_REGEX = r"^Rs\.\s?\d+(\.\d{2})?|₹\s?\d+(\.\d{2})?$"

UNIT_PRICE_TOLERANCE = 0.2  # 20% slack for OCR-noise when cross-checking unit price x quantity vs MRP


# --------------------------------------------------------------------------
# Typed ruleset models
# --------------------------------------------------------------------------

class RequiredField(BaseModel):
    """One field named in a Rule's required_fields, resolved against what we can actually check."""

    name: str
    has_validator: bool = False


class Rule(BaseModel):
    rule_id: str
    description: str
    status: str
    required_fields: List[str] = Field(default_factory=list)
    validations: Optional[Dict[str, Any]] = None
    logic: Optional[str] = None
    allowable_error: Optional[str] = None
    intervals: Optional[List[Dict[str, Any]]] = None
    validation_criteria: Optional[Dict[str, Any]] = None

    def required_field_objects(self) -> List[RequiredField]:
        return [
            RequiredField(name=name, has_validator=name in FIELD_VALIDATORS)
            for name in self.required_fields
        ]


class Framework(BaseModel):
    name: str
    scope: str
    rules: List[Rule]


class ValidationProfile(BaseModel):
    evaluate_rules: List[str] = Field(default_factory=list)
    fail_on_missing_field: Optional[str] = None


class Ruleset(BaseModel):
    ruleset_id: str
    title: str
    jurisdiction: str
    version: str
    last_updated: str
    frameworks: List[Framework]
    validation_profiles: Dict[str, ValidationProfile]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_default_ruleset() -> Ruleset:
    data = json.loads(DEFAULT_RULESET_PATH.read_text(encoding="utf-8"))
    return Ruleset.model_validate(data)


def load_ruleset(path: Optional[Path] = None) -> Ruleset:
    """Load and parse the Legal Metrology ruleset JSON into a typed Ruleset."""
    if path is None:
        return _load_default_ruleset()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Ruleset.model_validate(data)


def get_rule(ruleset: Ruleset, rule_id: str) -> Optional[Rule]:
    for framework in ruleset.frameworks:
        for rule in framework.rules:
            if rule.rule_id == rule_id:
                return rule
    return None


def get_profile(ruleset: Ruleset, profile_name: str) -> ValidationProfile:
    profile = ruleset.validation_profiles.get(profile_name)
    if profile is None:
        raise ValueError(
            f"Unknown validation profile {profile_name!r}. "
            f"Available: {sorted(ruleset.validation_profiles)}"
        )
    return profile


# --------------------------------------------------------------------------
# Field result
# --------------------------------------------------------------------------

class FieldResult(BaseModel):
    field: str
    status: Literal["present", "absent", "ambiguous"]
    extracted_value: Optional[str] = None
    compliant: Any  # bool | Literal["needs_review"], see field_validator below
    reason: Optional[str] = None
    conditional_field: bool = False


# --------------------------------------------------------------------------
# Shared regexes / lookup tables
# --------------------------------------------------------------------------

MANUFACTURER_PREFIX_RE = re.compile(
    r"(?:mfd\.?\s*by|manufactured\s+by|packed\s+by|packer\s*:|marketed\s+by|imported\s+by)\s*[:\-]?\s*(.{3,80})",
    re.IGNORECASE,
)

NET_QUANTITY_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(kgs?|grams?|gms?|gm|g|millilitres?|ml|litres?|liters?|ltr|l|pieces?|pcs?|units?|nos\.?)\b",
    re.IGNORECASE,
)

MONTH_NAMES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
MONTH_NAME_DATE_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?\s+(20\d{2})\b",
    re.IGNORECASE,
)
# Only 4-digit years, deliberately - a bare "MM/YY" is too easy to confuse with unrelated numbers.
NUMERIC_DATE_RE = re.compile(r"\b(0?[1-9]|1[0-2])[\/\-.](20\d{2})\b")

TAX_INCLUSIVE_RE = re.compile(r"incl(?:usive|\.)?\s*(?:of)?\s*(?:all)?\s*tax", re.IGNORECASE)

CURRENCY = r"(?:₹|Rs\.?)"
UNIT_PRICE_NUMERIC_RE = re.compile(
    CURRENCY + r"\s?(?P<price>\d+(?:\.\d+)?)\s*(?:/|per)\s*(?P<mult>\d+(?:\.\d+)?)?\s*"
    r"(?P<unit>kg|g|gram|gms?|gm|ml|l|litres?|liters?|ltr|piece|pieces|pc|pcs)\b",
    re.IGNORECASE,
)
UNIT_PRICE_LABEL_RE = re.compile(r"unit\s*sale\s*price|price\s*per\s*unit", re.IGNORECASE)

PHONE_RE = re.compile(
    r"(?:\+91[ -]?)?(?:1800[ -]?\d{3}[-\d ]{3,7}|\b[6-9]\d{9}\b|\b0\d{2,4}[ -]?\d{6,8}\b)"
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

COUNTRY_ORIGIN_RE = re.compile(
    r"(?:country\s+of\s+origin|made\s+in|origin)\s*[:\-]?\s*([A-Za-z][A-Za-z \-]{1,30})",
    re.IGNORECASE,
)

# category -> (unit -> multiplier into the category's base unit: grams for mass, ml for volume)
UNIT_BASE: Dict[str, Tuple[str, float]] = {
    "g": ("mass", 1.0), "gram": ("mass", 1.0), "grams": ("mass", 1.0), "gm": ("mass", 1.0), "gms": ("mass", 1.0),
    "kg": ("mass", 1000.0), "kgs": ("mass", 1000.0),
    "ml": ("volume", 1.0), "millilitre": ("volume", 1.0), "millilitres": ("volume", 1.0),
    "l": ("volume", 1000.0), "ltr": ("volume", 1000.0),
    "litre": ("volume", 1000.0), "litres": ("volume", 1000.0), "liter": ("volume", 1000.0), "liters": ("volume", 1000.0),
    "piece": ("count", 1.0), "pieces": ("count", 1.0), "pc": ("count", 1.0), "pcs": ("count", 1.0),
    "unit": ("count", 1.0), "units": ("count", 1.0), "nos": ("count", 1.0), "no": ("count", 1.0),
}


def _to_base(value: float, unit: str) -> Optional[Tuple[str, float]]:
    info = UNIT_BASE.get(unit.lower().rstrip("."))
    if not info:
        return None
    category, factor = info
    return category, value * factor


# --------------------------------------------------------------------------
# Field validators - one per Rule 6 mandatory declaration
# --------------------------------------------------------------------------

def check_manufacturer_packer_importer_details(text: str) -> FieldResult:
    field = "manufacturer_packer_importer_details"
    if not text or not text.strip():
        return FieldResult(
            field=field, status="absent", compliant=False,
            reason="No manufacturer/packer/importer declaration was found.",
        )
    match = MANUFACTURER_PREFIX_RE.search(text)
    if not match:
        return FieldResult(
            field=field, status="absent", compliant=False,
            reason="No manufacturer/packer/importer declaration (e.g. 'Mfd by', 'Packed by', "
                   "'Marketed by') was found.",
        )
    snippet = re.sub(r"\s+", " ", match.group(0)).strip()
    return FieldResult(field=field, status="present", extracted_value=snippet, compliant=True)


def check_common_generic_name(text: str) -> FieldResult:
    field = "common_generic_name_of_commodity"
    if not text or not text.strip():
        return FieldResult(
            field=field, status="absent", compliant=False,
            reason="No text was detected at all, so the commodity's common/generic name cannot be present.",
        )
    return FieldResult(
        field=field, status="ambiguous", compliant="needs_review",
        reason="The common/generic name of the commodity cannot be reliably isolated from raw OCR "
               "text without product-category matching. This is a weak heuristic pending a real "
               "field-detection step, so it's flagged for manual review rather than auto-passed or "
               "auto-failed.",
    )


def check_net_quantity(text: str) -> FieldResult:
    field = "net_quantity"
    if not text or not text.strip():
        return FieldResult(
            field=field, status="absent", compliant=False,
            reason="No net quantity declaration with a recognized unit was found.",
        )
    match = NET_QUANTITY_RE.search(text)
    if not match:
        return FieldResult(
            field=field, status="absent", compliant=False,
            reason="No numeric net quantity with a recognized unit (g, kg, ml, l, pieces, etc.) was found.",
        )
    return FieldResult(field=field, status="present", extracted_value=match.group(0).strip(), compliant=True)


def _parse_manufacture_date(text: str) -> Optional[Tuple[int, int, str]]:
    """Return (year, month, matched_text) for the first recognizable date, else None."""
    match = MONTH_NAME_DATE_RE.search(text)
    if match:
        month = MONTH_NAMES.get(match.group(1)[:3].lower())
        if month:
            return int(match.group(2)), month, match.group(0)
    match = NUMERIC_DATE_RE.search(text)
    if match:
        return int(match.group(2)), int(match.group(1)), match.group(0)
    return None


def check_manufacture_date(text: str) -> FieldResult:
    field = "month_and_year_of_manufacture_or_packing"
    if not text or not text.strip():
        return FieldResult(
            field=field, status="absent", compliant=False,
            reason="No manufacturing/packing month and year could be detected.",
        )
    parsed = _parse_manufacture_date(text)
    if not parsed:
        return FieldResult(
            field=field, status="absent", compliant=False,
            reason="No manufacturing/packing month and year could be detected in a recognized format "
                   "(e.g. 'MM/YYYY', 'MFG MAR 2026').",
        )
    year, month, matched_text = parsed
    now = datetime.now(timezone.utc)
    if (year, month) > (now.year, now.month):
        return FieldResult(
            field=field, status="present", extracted_value=matched_text, compliant=False,
            reason=f"Detected manufacturing/packing date ({matched_text!r}) is in the future, which "
                   "is not plausible.",
        )
    return FieldResult(field=field, status="present", extracted_value=matched_text, compliant=True)


@lru_cache(maxsize=4)
def _compile_mrp_pattern(regex_str: str) -> re.Pattern:
    return re.compile(regex_str, re.IGNORECASE | re.MULTILINE)


def _mrp_pattern(ruleset: Ruleset) -> re.Pattern:
    rule = get_rule(ruleset, "LMPC-R6-MANDATORY-DECLARATIONS")
    regex_str = FALLBACK_MRP_REGEX
    if rule and isinstance(rule.validations, dict):
        mrp_validation = rule.validations.get("mrp_format")
        if isinstance(mrp_validation, dict) and mrp_validation.get("regex"):
            regex_str = mrp_validation["regex"]
    return _compile_mrp_pattern(regex_str)


def check_mrp(text: str, ruleset: Optional[Ruleset] = None) -> FieldResult:
    """
    Uses the mrp_format regex from legal_metrology_rules.json directly (compiled with
    MULTILINE so ^/$ apply per line rather than to the whole text - the regex as written
    anchors each alternative to a line boundary, which only matches a line that IS the
    price, e.g. "Rs. 60.00" on its own OCR-detected line, not "MRP: Rs. 60.00" as one
    line. That's a real quirk inherited from the ruleset JSON, not something fixed here.
    """
    field = "maximum_retail_price_mrp"
    ruleset = ruleset or load_ruleset()
    if not text or not text.strip():
        return FieldResult(field=field, status="absent", compliant=False,
                            reason="No MRP declaration was found.")
    pattern = _mrp_pattern(ruleset)
    match = pattern.search(text)
    if not match:
        return FieldResult(
            field=field, status="absent", compliant=False,
            reason="No MRP value matching the expected currency format (e.g. 'Rs. 199.00' or "
                   "'₹199') was found.",
        )
    window = text[max(0, match.start() - 60): min(len(text), match.end() + 80)]
    if TAX_INCLUSIVE_RE.search(window):
        return FieldResult(field=field, status="present", extracted_value=match.group(0).strip(), compliant=True)
    return FieldResult(
        field=field, status="present", extracted_value=match.group(0).strip(), compliant="needs_review",
        reason="MRP value detected, but wording confirming it is 'inclusive of all taxes' was not "
               "found nearby; OCR may have missed it - verify manually.",
    )


def _parse_mrp_number(extracted_value: Optional[str]) -> Optional[float]:
    if not extracted_value:
        return None
    match = re.search(r"(\d+(?:\.\d{1,2})?)", extracted_value)
    return float(match.group(1)) if match else None


def _parse_quantity_number(extracted_value: Optional[str]) -> Optional[Tuple[float, str]]:
    if not extracted_value:
        return None
    match = NET_QUANTITY_RE.search(extracted_value)
    if not match:
        return None
    return float(match.group(1)), match.group(2)


def check_unit_sale_price(
    text: str,
    net_quantity_result: Optional[FieldResult] = None,
    mrp_result: Optional[FieldResult] = None,
) -> FieldResult:
    """
    Presence check for a per-unit price (e.g. '₹40/100g'). If net_quantity_result and
    mrp_result are also supplied and both look usable, cross-checks unit_price x quantity
    against the declared MRP. A mismatch is always "needs_review", never a hard fail - OCR
    misreads on any of the three numbers are common enough that auto-failing would be noisy.
    """
    field = "unit_sale_price"
    if not text or not text.strip():
        return FieldResult(field=field, status="absent", compliant=False,
                            reason="No unit sale price (price per unit quantity) was found.")

    numeric_match = UNIT_PRICE_NUMERIC_RE.search(text)
    label_match = None if numeric_match else UNIT_PRICE_LABEL_RE.search(text)

    if not numeric_match and not label_match:
        return FieldResult(
            field=field, status="absent", compliant=False,
            reason="No unit sale price (price per unit quantity, e.g. '₹40/100g') was found.",
        )

    if label_match and not numeric_match:
        return FieldResult(
            field=field, status="present", extracted_value=label_match.group(0).strip(), compliant="needs_review",
            reason="A unit-sale-price label was found but no parseable per-unit amount, so it "
                   "could not be cross-checked against MRP and net quantity - verify manually.",
        )

    extracted = numeric_match.group(0).strip()
    price = float(numeric_match.group("price"))
    mult = float(numeric_match.group("mult")) if numeric_match.group("mult") else 1.0
    price_base = _to_base(mult, numeric_match.group("unit"))

    if (
        not price_base
        or not net_quantity_result
        or not mrp_result
        or net_quantity_result.compliant is not True
        or mrp_result.compliant not in (True, "needs_review")
    ):
        return FieldResult(field=field, status="present", extracted_value=extracted, compliant=True)

    qty_parsed = _parse_quantity_number(net_quantity_result.extracted_value)
    mrp_value = _parse_mrp_number(mrp_result.extracted_value)
    if not qty_parsed or not mrp_value:
        return FieldResult(field=field, status="present", extracted_value=extracted, compliant=True)

    qty_base = _to_base(*qty_parsed)
    price_category, price_base_amount = price_base
    if not qty_base or qty_base[0] != price_category:
        return FieldResult(field=field, status="present", extracted_value=extracted, compliant=True)

    expected_total = (price / price_base_amount) * qty_base[1]
    deviation = abs(expected_total - mrp_value) / mrp_value
    if deviation <= UNIT_PRICE_TOLERANCE:
        return FieldResult(field=field, status="present", extracted_value=extracted, compliant=True)

    return FieldResult(
        field=field, status="present", extracted_value=extracted, compliant="needs_review",
        reason=(
            f"Unit price implies a total of ~₹{expected_total:.2f} for the declared net quantity, "
            f"which doesn't closely match the declared MRP (₹{mrp_value:.2f}); likely an OCR "
            "misread on one of the three values - verify manually rather than auto-failing."
        ),
    )


def _consumer_care_fields(ruleset: Ruleset) -> List[str]:
    rule = get_rule(ruleset, "LMPC-R6-MANDATORY-DECLARATIONS")
    if rule and isinstance(rule.validations, dict):
        fields = rule.validations.get("consumer_care_fields")
        if isinstance(fields, list):
            return [str(f) for f in fields]
    return ["name", "address", "telephone_number", "email_address"]


def check_consumer_care_details(text: str, ruleset: Optional[Ruleset] = None) -> FieldResult:
    field = "consumer_care_details"
    ruleset = ruleset or load_ruleset()
    expected_fields = ", ".join(_consumer_care_fields(ruleset))
    if not text or not text.strip():
        return FieldResult(field=field, status="absent", compliant=False,
                            reason="No consumer-care phone number or email address was found.")
    phone_match = PHONE_RE.search(text)
    email_match = EMAIL_RE.search(text)
    if not phone_match and not email_match:
        return FieldResult(field=field, status="absent", compliant=False,
                            reason="No consumer-care phone number or email address was found.")
    parts = [m.group(0).strip() for m in (phone_match, email_match) if m]
    extracted = " / ".join(parts)
    if phone_match and email_match:
        return FieldResult(field=field, status="present", extracted_value=extracted, compliant=True)
    return FieldResult(
        field=field, status="present", extracted_value=extracted, compliant="needs_review",
        reason=f"Only a phone number or an email was found, not both; Rule 6 expects full consumer-care "
               f"contact details ({expected_fields}) - verify the rest wasn't missed by OCR.",
    )


def check_country_of_origin(text: str) -> FieldResult:
    field = "country_of_origin"
    if text and text.strip():
        match = COUNTRY_ORIGIN_RE.search(text)
        if match:
            return FieldResult(
                field=field, status="present", extracted_value=match.group(0).strip(),
                compliant=True, conditional_field=True,
            )
    return FieldResult(
        field=field, status="absent", compliant=True, conditional_field=True,
        reason="Country of origin was not detected. This field is only mandatory for imported "
               "products, so its absence is not treated as a violation by default.",
    )


FIELD_VALIDATORS: Dict[str, Callable[..., FieldResult]] = {
    "manufacturer_packer_importer_details": check_manufacturer_packer_importer_details,
    "common_generic_name_of_commodity": check_common_generic_name,
    "net_quantity": check_net_quantity,
    "month_and_year_of_manufacture_or_packing": check_manufacture_date,
    "maximum_retail_price_mrp": check_mrp,
    "unit_sale_price": check_unit_sale_price,
    "consumer_care_details": check_consumer_care_details,
    "country_of_origin": check_country_of_origin,
}


# --------------------------------------------------------------------------
# Exemptions (stub)
# --------------------------------------------------------------------------

def check_exemptions(
    weight_kg: Optional[float] = None,
    category: Optional[str] = None,
    product_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Stub exemption check for Legal Metrology (Packaged Commodities) Rules exemptions
    (e.g. small-package or agricultural-produce exemptions). We don't yet capture package
    weight or product category anywhere upstream, so this always reports "no exemptions
    apply". The signature already accepts weight_kg/category/product_context so a real
    exemption ruleset can be wired in later without changing callers.
    """
    return {
        "exempt": False,
        "applicable_exemptions": [],
        "reason": (
            "No package weight or product category data is available yet, so exemption "
            "status cannot be determined; assuming standard Rule 6 declarations apply."
        ),
    }


# --------------------------------------------------------------------------
# Top-level orchestration
# --------------------------------------------------------------------------

def run_compliance_check(
    ocr_text: str,
    profile: str = "e_commerce_product_listing",
    ruleset: Optional[Ruleset] = None,
) -> Dict[str, Any]:
    """
    Run every field validator required by `profile` against `ocr_text` and return a
    report: {profile, ruleset_id, ruleset_version, evaluated_rules, fields, violations,
    exemptions, overall_status, notes}.

    overall_status is "Non-Compliant" if any non-conditional field hard-fails,
    "Partially Compliant" if nothing hard-fails but something needs_review, else "Compliant".
    """
    ruleset = ruleset or load_ruleset()
    validation_profile = get_profile(ruleset, profile)

    required_fields: List[str] = []
    for rule_id in validation_profile.evaluate_rules:
        rule = get_rule(ruleset, rule_id)
        if rule:
            for name in rule.required_fields:
                if name not in required_fields:
                    required_fields.append(name)

    notes: List[str] = []
    if not required_fields:
        notes.append(
            f"Profile {profile!r} has no OCR-checkable required fields (its rules concern "
            "physical instrument audits or other non-label criteria, not label text)."
        )

    field_results: Dict[str, FieldResult] = {}
    for name in required_fields:
        if name == "unit_sale_price":
            continue  # evaluated after net_quantity/mrp so it can cross-check them
        validator = FIELD_VALIDATORS.get(name)
        if validator is None:
            notes.append(f"No validator implemented yet for required field {name!r}.")
            continue
        kwargs: Dict[str, Any] = {}
        if name in ("maximum_retail_price_mrp", "consumer_care_details"):
            kwargs["ruleset"] = ruleset
        field_results[name] = validator(ocr_text, **kwargs)

    if "unit_sale_price" in required_fields:
        field_results["unit_sale_price"] = check_unit_sale_price(
            ocr_text,
            net_quantity_result=field_results.get("net_quantity"),
            mrp_result=field_results.get("maximum_retail_price_mrp"),
        )

    ordered_results = [field_results[name] for name in required_fields if name in field_results]

    violations = [
        {"field": r.field, "status": r.status, "compliant": r.compliant, "reason": r.reason}
        for r in ordered_results
        if r.compliant is not True
    ]

    hard_fail = any(r.compliant is False and not r.conditional_field for r in ordered_results)
    needs_review = any(r.compliant == "needs_review" for r in ordered_results)
    if hard_fail:
        overall_status = "Non-Compliant"
    elif needs_review:
        overall_status = "Partially Compliant"
    else:
        overall_status = "Compliant"

    return {
        "profile": profile,
        "ruleset_id": ruleset.ruleset_id,
        "ruleset_version": ruleset.version,
        "evaluated_rules": list(validation_profile.evaluate_rules),
        "fields": [r.model_dump() for r in ordered_results],
        "violations": violations,
        "exemptions": check_exemptions(),
        "overall_status": overall_status,
        "notes": notes,
    }
