"""
Phase 2: rule-based/keyword field classification, upstream of rule_engine.py.

Takes raw OCR output (full_text + a list of detections, each with text/confidence/bbox
- structurally matching backend/ocr/schemas.py's OCRResult/Detection, though this module
never imports that schema to stay self-contained: it accepts plain dicts or DetectionInput
instances) and attempts to work out which detection(s) correspond to each of the 8 Rule 6
fields, BEFORE rule_engine.py's validators run.

No ML/NER model - deliberately keyword + bounding-box-proximity heuristics only, per the
roadmap's stated preference to avoid a Python NER dependency for now. Two field categories:

  - "fused" fields (manufacturer_packer_importer_details, country_of_origin): a single
    rule_engine regex already captures keyword + value together (e.g. "Mfd by ..."), so a
    hit anywhere is confidence="high".
  - "keyword + value" fields (net_quantity, month_and_year_of_manufacture_or_packing,
    maximum_retail_price_mrp, unit_sale_price, consumer_care_details): a keyword anchor
    (e.g. "Net Wt", "MRP") is looked for in each detection in reading order; if the same
    detection also matches the field's value pattern, confidence="high"; if the value
    pattern is found in one of the next couple of detections instead (e.g. "MRP" and
    "Rs. 60.00" as two separate OCR-detected lines), confidence="medium"; if only a bare
    value pattern is found with no keyword anywhere nearby, confidence="low".

  - common_generic_name_of_commodity has no reliable keyword anchor at all (there's no
    standard "Product Name:" label on most Indian packaged-commodity labels) and is never
    classified here - it's left for rule_engine's own honest needs_review fallback.

Zero dependency on the FastAPI app; only depends on rule_engine.py within this same
package (for the shared regexes/types and the ruleset loader), keeping the coupling
one-directional (rule_engine never imports this module).
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from backend.compliance.rule_engine import (
    COUNTRY_ORIGIN_RE,
    EMAIL_RE,
    MANUFACTURER_PREFIX_RE,
    MONTH_NAME_DATE_RE,
    MONTH_NAMES,
    NET_QUANTITY_RE,
    NUMERIC_DATE_RE,
    PHONE_RE,
    UNIT_PRICE_NUMERIC_RE,
    ClassifiedDetection,
    ClassifiedField,
    ClassifiedFields,
    Ruleset,
    load_ruleset,
    mrp_pattern,
    to_base_amount,
)

# A manufacture/packing-date value-matcher must not grab an expiry/best-before date that
# happens to sit in the same or a neighbouring detection.
EXPIRY_NEGATIVE_RE = re.compile(r"best\s+before|expir|exp\.?\s*date|use\s+by", re.IGNORECASE)

# A net-quantity value-matcher must not grab a gram/ml figure from the Nutrition
# Information table (Energy/Protein/Carbohydrate/Fat per-100g values use the exact same
# "<number> g" shape as a real net-quantity declaration) that happens to sit in the same
# or a neighbouring detection.
NUTRITION_TABLE_NEGATIVE_RE = re.compile(
    r"nutrition(?:al)?\s*(?:information|facts|value)?|energy|protein|carbohydrate|"
    r"\btotal\s+fat\b|\bfat\b|per\s*100\s*g|approximate\s+value",
    re.IGNORECASE,
)

LOOKAHEAD_WINDOW = 2  # how many following detections count as "nearby" for a keyword-only anchor
PROXIMITY_WINDOW = 2  # how many detections before/after count as "nearby" for an exclusion check


def _context_matches(index: int, ordered: "List[DetectionInput]", pattern: "re.Pattern", window: int) -> bool:
    """
    True if `pattern` matches the detection at `index` itself, or any detection within
    `window` positions before/after it in reading order. Used to exclude a bare value
    (e.g. a date, or a "75 g" figure) from a field when a disqualifying keyword sits on
    a NEIGHBOURING OCR line rather than the same one - e.g. "USE BY" on one line and its
    date on the very next, which a same-text-only check would miss entirely.
    """
    lo = max(0, index - window)
    hi = min(len(ordered), index + window + 1)
    return any(pattern.search(ordered[j].text) for j in range(lo, hi))


def _has_own_keyword(text: str, keywords: "List[str]") -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


class DetectionInput(BaseModel):
    """Minimal detection shape this module needs - duck-type compatible with backend.ocr.schemas.Detection."""

    text: str
    confidence: float = 1.0
    bbox: List[int] = Field(default_factory=list)


def _normalize(detections: List[Union[DetectionInput, Dict[str, Any], Any]]) -> List[DetectionInput]:
    normalized: List[DetectionInput] = []
    for det in detections:
        if isinstance(det, DetectionInput):
            normalized.append(det)
        elif isinstance(det, dict):
            normalized.append(DetectionInput.model_validate(det))
        else:
            # duck-typed object (e.g. backend.ocr.schemas.Detection) - read attributes, don't import the type
            normalized.append(DetectionInput(
                text=getattr(det, "text", ""),
                confidence=getattr(det, "confidence", 1.0),
                bbox=list(getattr(det, "bbox", []) or []),
            ))
    return normalized


def _reading_order(detections: List[DetectionInput]) -> List[DetectionInput]:
    """Top-to-bottom, left-to-right by bbox when available; stable no-op otherwise."""
    def key(det: DetectionInput):
        if len(det.bbox) == 4:
            return (det.bbox[1], det.bbox[0])
        return (0, 0)  # stable sort preserves original order for detections without a usable bbox

    return sorted(detections, key=key)


def _to_classified_detection(det: DetectionInput) -> ClassifiedDetection:
    return ClassifiedDetection(text=det.text, confidence=det.confidence, bbox=det.bbox)


# --------------------------------------------------------------------------
# Field-specific keyword lists and value matchers
# --------------------------------------------------------------------------

FIELD_KEYWORDS: Dict[str, List[str]] = {
    "net_quantity": ["net wt", "net qty", "net weight", "net quantity", "contents"],
    "month_and_year_of_manufacture_or_packing": [
        "mfg", "mfd", "manufacturing date", "packing date", "pkd", "date of mfg", "date of packing",
    ],
    "maximum_retail_price_mrp": ["mrp", "maximum retail price", "m.r.p"],
    "unit_sale_price": ["unit sale price", "price per", "per kg", "per 100g", "usp", "per unit"],
    "consumer_care_details": [
        "consumer care", "customer care", "complaint", "contact us", "toll free", "helpline",
        "customer support",
    ],
}

ValueMatcher = Callable[[str], bool]


def _date_value_matches(text: str) -> bool:
    if EXPIRY_NEGATIVE_RE.search(text):
        return False  # this looks like an expiry/best-before line, not a manufacture date
    return bool(MONTH_NAME_DATE_RE.search(text) or NUMERIC_DATE_RE.search(text))


def _consumer_care_value_matches(text: str) -> bool:
    return bool(PHONE_RE.search(text) or EMAIL_RE.search(text))


def _value_matchers(ruleset: Ruleset) -> Dict[str, ValueMatcher]:
    mrp_re = mrp_pattern(ruleset)
    return {
        "net_quantity": lambda t: bool(NET_QUANTITY_RE.search(t)),
        "month_and_year_of_manufacture_or_packing": _date_value_matches,
        "maximum_retail_price_mrp": lambda t: bool(mrp_re.search(t)),
        # Deliberately NUMERIC-only, not UNIT_PRICE_LABEL_RE too: a bare "Unit Sale Price"
        # label with no number is a weak/no-op match, not a real value. If this matcher
        # accepted the label alone, a detection like "Unit Sale Price" (no number) would
        # satisfy the same-detection branch below and the classifier would stop right
        # there, never looking ahead to find the actual "Rs 30 per 100g" on the next line
        # - throwing away real data it should have combined. check_unit_sale_price itself
        # still falls back to UNIT_PRICE_LABEL_RE for a "found but unparseable" needs_review
        # result when no number is found anywhere nearby; that's unaffected by this.
        "unit_sale_price": lambda t: bool(UNIT_PRICE_NUMERIC_RE.search(t)),
        "consumer_care_details": _consumer_care_value_matches,
    }


# "Fused" fields: keyword + value already captured by one regex in rule_engine.py.
FUSED_FIELD_MATCHERS: Dict[str, ValueMatcher] = {
    "manufacturer_packer_importer_details": lambda t: bool(MANUFACTURER_PREFIX_RE.search(t)),
    "country_of_origin": lambda t: bool(COUNTRY_ORIGIN_RE.search(t)),
}


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def _classify_fused_field(field_name: str, matcher: ValueMatcher, ordered: List[DetectionInput]) -> ClassifiedField:
    for det in ordered:
        if matcher(det.text):
            return ClassifiedField(
                field=field_name,
                matched_detections=[_to_classified_detection(det)],
                combined_text=det.text,
                confidence="high",
            )
    return ClassifiedField(field=field_name)


def _dedup_detections(detections: List[DetectionInput]) -> List[DetectionInput]:
    seen = set()
    unique: List[DetectionInput] = []
    for det in detections:
        key = (det.text, tuple(det.bbox))
        if key not in seen:
            seen.add(key)
            unique.append(det)
    return unique


def _classify_consumer_care(keywords: List[str], ordered: List[DetectionInput]) -> ClassifiedField:
    """
    consumer_care_details needs its own logic: phone and email routinely land on two
    separate OCR detections (e.g. "Consumer Care: 1800-123-4567" then, as its own line,
    "Email: care@brand.example.com"), neither of which mentions the other. The generic
    keyword_value classifier stops at the first detection satisfying the value matcher,
    which would grab only the phone and silently drop the email sitting right next to it
    - worse than not narrowing the text at all. This gathers every phone/email hit found
    in the anchor detection plus a short lookahead window instead of stopping at the first.
    """
    field_name = "consumer_care_details"
    for i, det in enumerate(ordered):
        lower = det.text.lower()
        matched_keyword = next((kw for kw in keywords if kw in lower), None)
        if not matched_keyword:
            continue

        window = [det] + ordered[i + 1: i + 1 + LOOKAHEAD_WINDOW]
        phone_hit = next((d for d in window if PHONE_RE.search(d.text)), None)
        email_hit = next((d for d in window if EMAIL_RE.search(d.text)), None)

        if not phone_hit and not email_hit:
            return ClassifiedField(
                field=field_name,
                matched_detections=[_to_classified_detection(det)],
                combined_text=det.text,
                confidence="low",
                anchor_keyword=matched_keyword,
            )

        contributing = _dedup_detections([d for d in (det, phone_hit, email_hit) if d is not None])
        return ClassifiedField(
            field=field_name,
            matched_detections=[_to_classified_detection(d) for d in contributing],
            combined_text="\n".join(d.text for d in contributing),
            confidence="high" if (phone_hit and email_hit) else "medium",
            anchor_keyword=matched_keyword,
        )

    # No keyword anywhere; fall back to a bare phone/email match anywhere (low confidence).
    for det in ordered:
        if PHONE_RE.search(det.text) or EMAIL_RE.search(det.text):
            return ClassifiedField(
                field=field_name,
                matched_detections=[_to_classified_detection(det)],
                combined_text=det.text,
                confidence="low",
            )
    return ClassifiedField(field=field_name)


def _classify_keyword_value_field(
    field_name: str,
    keywords: List[str],
    matcher: ValueMatcher,
    ordered: List[DetectionInput],
    exclude_pattern: Optional["re.Pattern"] = None,
    anchor_exclude_pattern: Optional["re.Pattern"] = None,
) -> ClassifiedField:
    """
    `exclude_pattern`, when given, disqualifies a BARE candidate (one with no keyword
    evidence of its own) from being this field's value if it (or a neighbour within
    PROXIMITY_WINDOW) matches - e.g. a date sitting near "USE BY" is never a
    manufacture-date candidate, and a gram figure sitting near "Protein"/"Energy" is
    never a net-quantity candidate, even when the disqualifying keyword is a separate
    OCR detection from the value itself. It never overrides a candidate that carries the
    field's OWN keyword - that's direct, positive evidence unrelated text nearby can't undo.

    `anchor_exclude_pattern`, when given, disqualifies a detection from being this
    field's keyword ANCHOR even though it contains one of `keywords` as a substring -
    e.g. "mfd" is a legitimate date keyword ("MFD: 12/2024"), but "Mfd by Britannia..."
    is the MANUFACTURER declaration, matched by MANUFACTURER_PREFIX_RE; without this, a
    naive substring match would misattribute the date field's anchor to that line instead
    of a real date-bearing one, and the ambiguity between fields is worse for the exact
    same reason keyword collisions are handled via `claimed_by` in conflict detection.
    """
    def excluded(index: int, candidate_text: str) -> bool:
        if _has_own_keyword(candidate_text, keywords):
            return False
        return bool(exclude_pattern and _context_matches(index, ordered, exclude_pattern, PROXIMITY_WINDOW))

    for i, det in enumerate(ordered):
        # No excluded(i) check here: det.text carrying this field's OWN keyword is direct,
        # positive evidence that must not be overridden just because unrelated text (e.g.
        # a Best Before line, or a nutrition-table row) happens to sit nearby. Exclusion
        # only matters for candidates that have no keyword evidence of their own - the
        # lookahead and bare-fallback checks below.
        lower = det.text.lower()
        matched_keyword = next((kw for kw in keywords if kw in lower), None)
        if not matched_keyword:
            continue
        if anchor_exclude_pattern and anchor_exclude_pattern.search(det.text):
            continue  # keyword substring matched, but this line belongs to a different field

        if matcher(det.text):
            return ClassifiedField(
                field=field_name,
                matched_detections=[_to_classified_detection(det)],
                combined_text=det.text,
                confidence="high",
                anchor_keyword=matched_keyword,
            )

        for j in range(i + 1, min(i + 1 + LOOKAHEAD_WINDOW, len(ordered))):
            candidate = ordered[j]
            if excluded(j, candidate.text):
                continue
            if matcher(candidate.text):
                return ClassifiedField(
                    field=field_name,
                    matched_detections=[_to_classified_detection(det), _to_classified_detection(candidate)],
                    combined_text=det.text + "\n" + candidate.text,
                    confidence="medium",
                    anchor_keyword=matched_keyword,
                )

        # Keyword found but no value nearby - still worth surfacing as a weak anchor.
        return ClassifiedField(
            field=field_name,
            matched_detections=[_to_classified_detection(det)],
            combined_text=det.text,
            confidence="low",
            anchor_keyword=matched_keyword,
        )

    # No keyword anywhere; fall back to a bare value match anywhere in the text (low confidence).
    for i, det in enumerate(ordered):
        if excluded(i, det.text):
            continue
        if matcher(det.text):
            return ClassifiedField(
                field=field_name,
                matched_detections=[_to_classified_detection(det)],
                combined_text=det.text,
                confidence="low",
            )

    return ClassifiedField(field=field_name)


# --------------------------------------------------------------------------
# Phase 6: multi-face / conflicting-value detection.
#
# A field's normal classification (above) always returns the FIRST matching
# detection - fine for a single-face label, but a multi-face package (or two
# printings that disagree) can have two genuinely different values for the same
# field, e.g. "Rs. 60.00" on one panel and "Rs. 65.00" on another. Silently
# keeping the first one would hide a real discrepancy. This pass re-scans every
# detection for the field's underlying value (never the keyword-anchored/lookahead
# result above), and if two or more DISTINCT normalized values are found, replaces
# that field's classification with an ambiguous one carrying every candidate.
#
# Deliberately scoped to fields where "genuinely different value" can be checked
# reliably by comparing parsed numbers/dates - not manufacturer_packer_importer_details
# or country_of_origin, where comparing free-text names for semantic equality isn't
# reliable with regex alone (OCR noise would create constant false alarms), and not
# common_generic_name_of_commodity (never classified at all) or consumer_care_details
# (multiple phone/email hits are complementary, not conflicting).
# --------------------------------------------------------------------------

# Normalizers are index-aware (index, detection, ordered) rather than just (text,) so
# they can apply the same PROXIMITY_WINDOW exclusion the primary classifier uses (see
# _classify_keyword_value_field) - a bare nutrition-table figure or expiry date sitting
# near its disqualifying keyword on a NEIGHBOURING detection must be excluded from the
# conflict scan too, not just from the keyword-anchored pass.
Normalizer = Callable[[int, DetectionInput, List[DetectionInput]], Optional[Any]]


def _normalize_net_quantity(index: int, det: DetectionInput, ordered: List[DetectionInput]) -> Optional[Any]:
    # A detection carrying "net wt"/"net qty"/etc itself is direct evidence and must not
    # be excluded just because an unrelated nutrition-table row sits nearby (e.g. a real
    # "Rs 30 per 100g" unit-price line would otherwise get excluded by its own "per 100g").
    if not _has_own_keyword(det.text, FIELD_KEYWORDS["net_quantity"]):
        if _context_matches(index, ordered, NUTRITION_TABLE_NEGATIVE_RE, PROXIMITY_WINDOW):
            return None  # a gram figure from the Nutrition Information table, not net quantity
    match = NET_QUANTITY_RE.search(det.text)
    if not match:
        return None
    base = to_base_amount(float(match.group(1)), match.group(2))
    return (base[0], round(base[1], 4)) if base else None


def _normalize_date(index: int, det: DetectionInput, ordered: List[DetectionInput]) -> Optional[Any]:
    # Same reasoning as _normalize_net_quantity: a detection with its own MFG-style
    # keyword is direct evidence, not overridden by an unrelated expiry line nearby.
    if not _has_own_keyword(det.text, FIELD_KEYWORDS["month_and_year_of_manufacture_or_packing"]):
        if _context_matches(index, ordered, EXPIRY_NEGATIVE_RE, PROXIMITY_WINDOW):
            return None  # never treat an expiry/best-before/use-by date as a manufacture-date candidate
    match = MONTH_NAME_DATE_RE.search(det.text)
    if match:
        month = MONTH_NAMES.get(match.group(1)[:3].lower())
        if month:
            return int(match.group(2)), month
    match = NUMERIC_DATE_RE.search(det.text)
    if match:
        return int(match.group(2)), int(match.group(1))
    return None


def _normalize_unit_price(index: int, det: DetectionInput, ordered: List[DetectionInput]) -> Optional[Any]:
    match = UNIT_PRICE_NUMERIC_RE.search(det.text)
    if not match:
        return None
    mult = float(match.group("mult")) if match.group("mult") else 1.0
    base = to_base_amount(mult, match.group("unit"))
    if not base:
        return None
    category, base_amount = base
    return category, round(float(match.group("price")) / base_amount, 6)


def _make_mrp_normalizer(ruleset: Ruleset) -> Normalizer:
    pattern = mrp_pattern(ruleset)

    def normalize(index: int, det: DetectionInput, ordered: List[DetectionInput]) -> Optional[float]:
        match = pattern.search(det.text)
        if not match:
            return None
        digits = re.search(r"(\d+(?:\.\d{1,2})?)", match.group(0))
        return round(float(digits.group(1)), 2) if digits else None

    return normalize


def _detection_key(det: DetectionInput) -> Any:
    return det.text, tuple(det.bbox)


def _detect_value_conflicts(
    field_name: str,
    normalizer: Normalizer,
    ordered: List[DetectionInput],
    claimed_by: Dict[Any, str],
) -> Optional[ClassifiedField]:
    """
    `claimed_by` maps a detection to whichever field's PRIMARY classification (the
    keyword-anchored pass above) already matched-detections it. A detection claimed by a
    DIFFERENT field is skipped here: value-pattern regexes can coincidentally collide
    across fields (e.g. "Rs 30 per 100g" contains "100g", which also matches the net
    quantity regex) - without this, that unit-price detection would look like a second,
    conflicting net_quantity value, which it isn't.
    """
    groups: Dict[Any, DetectionInput] = {}
    for i, det in enumerate(ordered):
        owner = claimed_by.get(_detection_key(det))
        if owner is not None and owner != field_name:
            continue
        key = normalizer(i, det, ordered)
        if key is not None and key not in groups:
            groups[key] = det  # keep the first detection seen for each distinct value

    if len(groups) <= 1:
        return None

    contributing = list(groups.values())
    return ClassifiedField(
        field=field_name,
        matched_detections=[_to_classified_detection(d) for d in contributing],
        combined_text="\n".join(d.text for d in contributing),
        ambiguous=True,
        candidate_values=[d.text.strip() for d in contributing],
    )


def classify_fields(
    full_text: str,
    detections: List[Union[DetectionInput, Dict[str, Any], Any]],
    ruleset: Optional[Ruleset] = None,
) -> ClassifiedFields:
    """
    Classify OCR detections against the 8 Rule 6 fields.

    `detections` accepts DetectionInput instances, plain dicts shaped like
    {"text": ..., "confidence": ..., "bbox": [x1,y1,x2,y2]}, or any object exposing those
    same attributes (e.g. backend.ocr.schemas.Detection) - the type is never imported here.
    """
    ruleset = ruleset or load_ruleset()
    ordered = _reading_order(_normalize(detections))
    matchers = _value_matchers(ruleset)

    fields: Dict[str, ClassifiedField] = {}
    for field_name, matcher in FUSED_FIELD_MATCHERS.items():
        fields[field_name] = _classify_fused_field(field_name, matcher, ordered)

    field_exclusions: Dict[str, "re.Pattern"] = {
        "month_and_year_of_manufacture_or_packing": EXPIRY_NEGATIVE_RE,
        "net_quantity": NUTRITION_TABLE_NEGATIVE_RE,
    }
    # "mfd"/"mfg" are legitimate date keywords ("MFD: 12/2024") but also substrings of
    # "Mfd by ..."/"Manufactured by ..." - the manufacturer declaration, not a date. Without
    # this, that line would wrongly anchor the date field instead of a real date-bearing one.
    anchor_exclusions: Dict[str, "re.Pattern"] = {
        "month_and_year_of_manufacture_or_packing": MANUFACTURER_PREFIX_RE,
    }
    for field_name, keywords in FIELD_KEYWORDS.items():
        if field_name == "consumer_care_details":
            fields[field_name] = _classify_consumer_care(keywords, ordered)
        else:
            fields[field_name] = _classify_keyword_value_field(
                field_name, keywords, matchers[field_name], ordered,
                exclude_pattern=field_exclusions.get(field_name),
                anchor_exclude_pattern=anchor_exclusions.get(field_name),
            )

    # common_generic_name_of_commodity is intentionally never classified here - see module docstring.
    fields["common_generic_name_of_commodity"] = ClassifiedField(field="common_generic_name_of_commodity")

    claimed_by: Dict[Any, str] = {}
    for fname, cf in fields.items():
        for md in cf.matched_detections:
            claimed_by[(md.text, tuple(md.bbox))] = fname

    conflict_normalizers: Dict[str, Normalizer] = {
        "net_quantity": _normalize_net_quantity,
        "month_and_year_of_manufacture_or_packing": _normalize_date,
        "maximum_retail_price_mrp": _make_mrp_normalizer(ruleset),
        "unit_sale_price": _normalize_unit_price,
    }
    for field_name, normalizer in conflict_normalizers.items():
        conflict = _detect_value_conflicts(field_name, normalizer, ordered, claimed_by)
        if conflict:
            fields[field_name] = conflict

    return ClassifiedFields(fields=fields, full_text=full_text)
