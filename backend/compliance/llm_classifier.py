"""
LLM-based field extraction, as a configurable alternative to field_classifier.py's
regex/keyword-proximity heuristics (see pipeline.py's COMPLIANCE_CLASSIFIER_MODE).

Uses the Groq API (openai/gpt-oss-120b - see _MODEL docstring below for why) to extract
the 8 Rule 6 fields from OCR text in one call, returning the SAME ClassifiedFields/
ClassifiedField shape field_classifier.classify_fields() produces, so rule_engine.py's
validators need no changes to consume either source.

Deliberately optional: if GROQ_API_KEY isn't set, or the `groq` package isn't installed,
or the API call fails for any reason (timeout, rate limit, malformed response), this
module raises LLMClassificationError rather than ever crashing the compliance check -
pipeline.py catches that and falls back to the regex classifier. See is_llm_available().
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

from backend.compliance.rule_engine import ClassifiedDetection, ClassifiedField, ClassifiedFields

logger = logging.getLogger("legallense.compliance.llm_classifier")

# openai/gpt-oss-120b: the largest general-purpose instruction-following model currently
# hosted on Groq (checked live against the Groq API's /models endpoint, not assumed from
# training data - Groq's lineup has shifted since training cutoff; llama-3.x models that
# used to be Groq's flagship are no longer listed). Chosen over the smaller gpt-oss-20b/
# qwen3.x-27b options for stronger structured-extraction reliability, and over
# groq/compound (an agentic tool-orchestration system, not a plain extraction model - the
# wrong shape for a single structured-JSON call). Supports response_format=json_object.
# Pricing per Groq's own docs (console.groq.com/docs/model/openai/gpt-oss-120b): $0.15 /
# 1M input tokens, $0.60 / 1M output tokens - no documented free-tier allocation, but at
# this price one compliance check costs a small fraction of a cent regardless.
MODEL = "openai/gpt-oss-120b"
PRICE_PER_1M_INPUT_TOKENS = 0.15
PRICE_PER_1M_OUTPUT_TOKENS = 0.60

REQUEST_TIMEOUT_SECONDS = 20

RULE6_FIELDS: List[str] = [
    "manufacturer_packer_importer_details",
    "country_of_origin",
    "common_generic_name_of_commodity",
    "net_quantity",
    "month_and_year_of_manufacture_or_packing",
    "maximum_retail_price_mrp",
    "unit_sale_price",
    "consumer_care_details",
]

FIELD_DESCRIPTIONS: Dict[str, str] = {
    "manufacturer_packer_importer_details": "Name and address of the manufacturer, packer, or importer (e.g. text after 'Mfd by', 'Packed by', 'Marketed by').",
    "country_of_origin": "The country the product was made in (only present on imported products - often absent, which is fine).",
    "common_generic_name_of_commodity": "The common/generic name of the product itself (e.g. 'Glucose Biscuits', 'Namkeen Mixture') - NOT the brand name alone.",
    "net_quantity": "The declared net weight/volume/count of the product itself (e.g. 'Net Wt 75 g'). Do NOT use figures from a Nutrition Information table (Energy/Protein/Carbohydrate/Fat per-100g values) - those are nutritional content, not net quantity.",
    "month_and_year_of_manufacture_or_packing": "The month and year the product was MANUFACTURED or PACKED (often labeled MFG, MFD, PKD, 'Manufacturing Date', 'Packing Date'). This is NOT the same as an expiry/best-before/use-by date - see expiry_or_use_by_date below, which is a separate concept entirely.",
    "maximum_retail_price_mrp": "The Maximum Retail Price, usually with a Rs./₹ currency symbol, often noted as inclusive of all taxes.",
    "unit_sale_price": "The price per unit quantity (e.g. '₹40 per 100g'), distinct from the MRP for the whole package.",
    "consumer_care_details": "Phone number and/or email address for consumer complaints/customer care.",
}


class LLMClassificationError(Exception):
    """Raised for any failure that should trigger pipeline.py's fallback to regex classification."""


class LLMUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int, elapsed_seconds: float):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.elapsed_seconds = elapsed_seconds
        self.estimated_cost_usd = (
            prompt_tokens / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
            + completion_tokens / 1_000_000 * PRICE_PER_1M_OUTPUT_TOKENS
        )

    def log(self) -> None:
        logger.info(
            "Groq %s call: %d prompt + %d completion = %d tokens, ~$%.6f, %.2fs",
            MODEL, self.prompt_tokens, self.completion_tokens, self.total_tokens,
            self.estimated_cost_usd, self.elapsed_seconds,
        )


def is_llm_available() -> bool:
    """True only if an API key is configured AND the groq package is importable."""
    if not os.getenv("GROQ_API_KEY", "").strip():
        return False
    try:
        import groq  # noqa: F401
    except ImportError:
        return False
    return True


def _build_system_prompt() -> str:
    field_lines = "\n".join(f'  - "{key}": {desc}' for key, desc in FIELD_DESCRIPTIONS.items())
    return f"""You extract specific fields from OCR text taken from a photograph of a packaged \
commodity label, for an Indian Legal Metrology (Packaged Commodities) Rules, 2011 compliance \
tool used by enforcement officials.

This is a LEGAL COMPLIANCE tool. A false positive (claiming a field is present with a made-up \
or wrong value) is much worse than admitting a field wasn't found. If the text for a field is \
genuinely not present in the provided OCR output, or you are not confident, return null for its \
value - do NOT guess, infer, or fabricate a plausible-sounding value that is not actually \
present verbatim (or near-verbatim, accounting for OCR noise) in the given text. Only use text \
that actually appears in the numbered detections provided.

The OCR input is a list of separately detected text lines (numbered), not clean prose - lines \
belonging to one logical field (e.g. a keyword label and its value) are sometimes split across \
two or more detections. Also be aware OCR text may contain misreads/typos.

Extract these 8 fields:
{field_lines}

Additionally, separately identify any expiry / best-before / use-by date under the key \
"expiry_or_use_by_date" - this is NOT one of the 8 fields above and must never be confused with \
or substituted for month_and_year_of_manufacture_or_packing. These are two different dates \
answering two different questions (when was it made vs when does it expire); extract them \
independently even if only one or neither is present.

Respond with ONLY a JSON object of this exact shape (no other text):
{{
  "fields": {{
    "<field_key>": {{
      "value": "<verbatim or near-verbatim extracted text, or null if not found>",
      "confidence": "high" | "medium" | "low",
      "detection_indices": [<int>, ...]
    }},
    ... one entry for each of the 8 field keys listed above ...
  }},
  "expiry_or_use_by_date": {{
    "value": "<extracted text, or null if not found>",
    "detection_indices": [<int>, ...]
  }}
}}

"confidence" and "detection_indices" must still be present even when value is null (use "low" \
and an empty list in that case). detection_indices must reference the indices given in the \
numbered detection list you're provided, identifying which detection(s) the value came from."""


def _build_user_message(detections: List[Dict[str, Any]]) -> str:
    lines = [f"[{i}] {det['text']!r}" for i, det in enumerate(detections)]
    return "Numbered OCR detections (index, then the detected text):\n" + "\n".join(lines)


def _normalize_detections(detections: List[Any]) -> List[Dict[str, Any]]:
    normalized = []
    for det in detections:
        if isinstance(det, dict):
            normalized.append({"text": det.get("text", ""), "bbox": det.get("bbox", []), "confidence": det.get("confidence", 1.0)})
        else:
            normalized.append({
                "text": getattr(det, "text", ""),
                "bbox": list(getattr(det, "bbox", []) or []),
                "confidence": getattr(det, "confidence", 1.0),
            })
    return normalized


def _field_result_to_classified_field(
    field_name: str, raw: Optional[Dict[str, Any]], detections: List[Dict[str, Any]]
) -> ClassifiedField:
    if not raw or not isinstance(raw, dict):
        return ClassifiedField(field=field_name)

    value = raw.get("value")
    if value in (None, "", "null"):
        return ClassifiedField(field=field_name)

    confidence = raw.get("confidence")
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    indices = raw.get("detection_indices") or []
    matched: List[ClassifiedDetection] = []
    for idx in indices:
        if isinstance(idx, int) and 0 <= idx < len(detections):
            d = detections[idx]
            matched.append(ClassifiedDetection(text=d["text"], confidence=d.get("confidence", 1.0), bbox=d.get("bbox", [])))

    combined_text = "\n".join(m.text for m in matched) if matched else str(value)

    return ClassifiedField(
        field=field_name,
        matched_detections=matched,
        combined_text=combined_text,
        confidence=confidence,
    )


def classify_fields_with_llm(
    ocr_text: str,
    detections: List[Any],
    model: str = MODEL,
) -> "tuple[ClassifiedFields, LLMUsage, Optional[str]]":
    """
    Extracts the 8 Rule 6 fields via the Groq API. Returns (classified_fields, usage,
    expiry_date_value) - expiry_date_value is informational only (see module docstring:
    not one of the 8 tracked fields, kept separate on purpose) and is not part of the
    returned ClassifiedFields.

    Raises LLMClassificationError on any failure (missing key, network/timeout/rate-limit,
    malformed response) - callers should catch this and fall back to classify_fields().
    """
    if not is_llm_available():
        raise LLMClassificationError("GROQ_API_KEY is not set or the groq package is not installed")

    import groq

    normalized = _normalize_detections(detections)
    client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"), timeout=REQUEST_TIMEOUT_SECONDS)

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": _build_user_message(normalized)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=2000,
        )
    except groq.GroqError as exc:
        raise LLMClassificationError(f"Groq API call failed: {exc}") from exc
    except Exception as exc:  # network errors etc. that aren't groq.GroqError subclasses
        raise LLMClassificationError(f"Groq API call failed: {exc}") from exc
    elapsed = time.time() - start

    usage_data = response.usage
    usage = LLMUsage(
        prompt_tokens=getattr(usage_data, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage_data, "completion_tokens", 0) or 0,
        elapsed_seconds=elapsed,
    )
    usage.log()

    raw_content = response.choices[0].message.content if response.choices else None
    if not raw_content:
        raise LLMClassificationError("Groq response had no content")

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise LLMClassificationError(f"Groq response was not valid JSON: {exc}") from exc

    raw_fields = parsed.get("fields")
    if not isinstance(raw_fields, dict):
        raise LLMClassificationError("Groq response missing a 'fields' object")

    fields: Dict[str, ClassifiedField] = {}
    for field_name in RULE6_FIELDS:
        fields[field_name] = _field_result_to_classified_field(field_name, raw_fields.get(field_name), normalized)

    expiry_raw = parsed.get("expiry_or_use_by_date")
    expiry_value = None
    if isinstance(expiry_raw, dict):
        v = expiry_raw.get("value")
        expiry_value = v if v not in (None, "", "null") else None

    classified_fields = ClassifiedFields(fields=fields, full_text=ocr_text)
    return classified_fields, usage, expiry_value
