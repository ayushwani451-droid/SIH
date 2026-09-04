"""
Tests for backend/compliance/llm_classifier.py.

LLM output isn't perfectly deterministic, so this file is split in two:
  - Deterministic tests (always run, no network/API key needed): prompt-building,
    response-parsing logic, and the no-key/unavailable fallback path, using a mocked
    Groq client so the main suite never makes a real network call or costs money.
  - One real, live smoke test against the actual Groq API, gated by
    @unittest.skipUnless(os.getenv("GROQ_API_KEY"), ...) - skipped in CI/normal runs
    without a key, but exercises the real integration when a key is available. It checks
    that expected fields are found with REASONABLE values (substring/keyword checks), not
    exact string matches, since real model output can vary slightly run to run.

Run with:
    backend\\.venv\\Scripts\\python.exe -m unittest backend.compliance.test_llm_classifier -v
"""
from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from backend.compliance.llm_classifier import (
    LLMClassificationError,
    _build_system_prompt,
    _build_user_message,
    _field_result_to_classified_field,
    classify_fields_with_llm,
    is_llm_available,
)
RULE6_FIELDS = [
    "manufacturer_packer_importer_details",
    "country_of_origin",
    "common_generic_name_of_commodity",
    "net_quantity",
    "month_and_year_of_manufacture_or_packing",
    "maximum_retail_price_mrp",
    "unit_sale_price",
    "consumer_care_details",
]


def _fake_groq_response(content: str, prompt_tokens=100, completion_tokens=50):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return resp


class AvailabilityTests(unittest.TestCase):
    def test_unavailable_without_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            self.assertFalse(is_llm_available())

    def test_available_with_api_key(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-test"}):
            self.assertTrue(is_llm_available())

    def test_classify_raises_clean_error_without_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            with self.assertRaises(LLMClassificationError):
                classify_fields_with_llm("some text", [])


class PromptBuildingTests(unittest.TestCase):
    def test_system_prompt_mentions_all_8_fields_and_anti_hallucination_instructions(self):
        prompt = _build_system_prompt()
        for field in RULE6_FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, prompt)
        self.assertIn("expiry_or_use_by_date", prompt)
        self.assertIn("do NOT guess", prompt.replace("do not guess", "do NOT guess"))
        lowered = prompt.lower()
        self.assertIn("null", lowered)
        self.assertIn("legal", lowered)

    def test_user_message_numbers_detections(self):
        detections = [{"text": "MRP Rs 60", "bbox": []}, {"text": "Net Qty 200 g", "bbox": []}]
        msg = _build_user_message(detections)
        self.assertIn("[0]", msg)
        self.assertIn("[1]", msg)
        self.assertIn("MRP Rs 60", msg)
        self.assertIn("Net Qty 200 g", msg)


class ResponseParsingTests(unittest.TestCase):
    def setUp(self):
        self.detections = [
            {"text": "Net Qty 200 g", "bbox": [10, 10, 100, 30], "confidence": 0.9},
            {"text": "MRP Rs. 60.00", "bbox": [10, 32, 100, 52], "confidence": 0.9},
        ]

    def test_null_value_produces_empty_classified_field(self):
        field = _field_result_to_classified_field("x", {"value": None, "confidence": "low", "detection_indices": []}, self.detections)
        self.assertIsNone(field.confidence)
        self.assertEqual(field.matched_detections, [])

    def test_missing_raw_produces_empty_classified_field(self):
        field = _field_result_to_classified_field("x", None, self.detections)
        self.assertIsNone(field.confidence)

    def test_valid_value_with_indices_resolves_real_detections(self):
        raw = {"value": "200 g", "confidence": "high", "detection_indices": [0]}
        field = _field_result_to_classified_field("net_quantity", raw, self.detections)
        self.assertEqual(field.confidence, "high")
        self.assertEqual(len(field.matched_detections), 1)
        self.assertEqual(field.matched_detections[0].bbox, [10, 10, 100, 30])

    def test_out_of_range_index_is_ignored_not_crashed_on(self):
        raw = {"value": "200 g", "confidence": "high", "detection_indices": [99, -1, 0]}
        field = _field_result_to_classified_field("net_quantity", raw, self.detections)
        self.assertEqual(len(field.matched_detections), 1)  # only index 0 is valid

    def test_invalid_confidence_defaults_to_medium(self):
        raw = {"value": "200 g", "confidence": "extremely sure", "detection_indices": [0]}
        field = _field_result_to_classified_field("net_quantity", raw, self.detections)
        self.assertEqual(field.confidence, "medium")

    def test_multiple_indices_combine_text(self):
        raw = {"value": "irrelevant", "confidence": "high", "detection_indices": [0, 1]}
        field = _field_result_to_classified_field("x", raw, self.detections)
        self.assertIn("Net Qty 200 g", field.combined_text)
        self.assertIn("MRP Rs. 60.00", field.combined_text)


class MockedFullCallTests(unittest.TestCase):
    """Exercises classify_fields_with_llm() end to end with a mocked Groq client - no
    network call, deterministic, verifies parsing/usage-tracking against a realistic
    fake response shape."""

    def _mock_response_json(self):
        fields = {}
        for f in RULE6_FIELDS:
            fields[f] = {"value": None, "confidence": "low", "detection_indices": []}
        fields["net_quantity"] = {"value": "200 g", "confidence": "high", "detection_indices": [0]}
        return json.dumps({
            "fields": fields,
            "expiry_or_use_by_date": {"value": "11/2026", "detection_indices": []},
        })

    @patch("groq.Groq")
    def test_successful_call_returns_classified_fields_and_usage(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_groq_response(
            self._mock_response_json(), prompt_tokens=500, completion_tokens=200,
        )
        mock_groq_cls.return_value = mock_client

        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-test"}):
            classified, usage, expiry = classify_fields_with_llm(
                "Net Qty 200 g", [{"text": "Net Qty 200 g", "bbox": [1, 2, 3, 4], "confidence": 0.9}],
            )

        self.assertEqual(classified.get("net_quantity").confidence, "high")
        self.assertIsNone(classified.get("maximum_retail_price_mrp").confidence)
        self.assertEqual(expiry, "11/2026")
        self.assertEqual(usage.prompt_tokens, 500)
        self.assertEqual(usage.completion_tokens, 200)
        self.assertGreater(usage.estimated_cost_usd, 0)

    @patch("groq.Groq")
    def test_malformed_json_response_raises_llm_classification_error(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_groq_response("not valid json{{{")
        mock_groq_cls.return_value = mock_client

        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-test"}):
            with self.assertRaises(LLMClassificationError):
                classify_fields_with_llm("text", [])

    @patch("groq.Groq")
    def test_missing_fields_key_raises_llm_classification_error(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_groq_response(json.dumps({"oops": "no fields key"}))
        mock_groq_cls.return_value = mock_client

        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-test"}):
            with self.assertRaises(LLMClassificationError):
                classify_fields_with_llm("text", [])

    @patch("groq.Groq")
    def test_api_exception_is_wrapped_as_llm_classification_error(self, mock_groq_cls):
        import groq as groq_module

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = groq_module.APITimeoutError(request=MagicMock())
        mock_groq_cls.return_value = mock_client

        with patch.dict(os.environ, {"GROQ_API_KEY": "fake-key-for-test"}):
            with self.assertRaises(LLMClassificationError):
                classify_fields_with_llm("text", [])


@unittest.skipUnless(os.getenv("GROQ_API_KEY"), "requires a real GROQ_API_KEY - skipped without one")
class LiveGroqSmokeTest(unittest.TestCase):
    """Real network call against the actual Groq API. Only runs when a real key is
    configured. Checked with reasonable/substring assertions, not exact string matches,
    since live model output isn't perfectly deterministic run to run."""

    def test_real_call_extracts_obvious_fields_from_simple_text(self):
        detections = [
            {"text": "Net Qty 200 g", "bbox": [10, 10, 100, 30], "confidence": 0.95},
            {"text": "MRP Rs. 60.00 Inclusive of all taxes", "bbox": [10, 32, 200, 52], "confidence": 0.95},
            {"text": "Mfd by Acme Foods Pvt Ltd, Delhi", "bbox": [10, 54, 300, 74], "confidence": 0.95},
        ]
        full_text = "\n".join(d["text"] for d in detections)
        classified, usage, expiry = classify_fields_with_llm(full_text, detections)

        net_qty = classified.get("net_quantity")
        self.assertIsNotNone(net_qty.confidence)
        self.assertIn("200", net_qty.combined_text)

        mrp = classified.get("maximum_retail_price_mrp")
        self.assertIsNotNone(mrp.confidence)
        self.assertIn("60", mrp.combined_text)

        manufacturer = classified.get("manufacturer_packer_importer_details")
        self.assertIsNotNone(manufacturer.confidence)
        self.assertIn("Acme", manufacturer.combined_text)

        self.assertGreater(usage.total_tokens, 0)
        self.assertGreater(usage.estimated_cost_usd, 0)


if __name__ == "__main__":
    unittest.main()
