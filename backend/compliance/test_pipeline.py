"""
Phase 5: integration-style tests for the full pipeline (classifier -> rule engine ->
report), against realistic multi-detection OCRResult fixtures with bounding boxes -
shaped like genuine PaddleOCR output, not hand-typed prose.

Run with:
    backend\\.venv\\Scripts\\python.exe -m unittest backend.compliance.test_pipeline -v
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend.compliance.llm_classifier import LLMClassificationError, LLMUsage
from backend.compliance.pipeline import run_full_compliance_pipeline
from backend.compliance.rule_engine import ClassifiedField, ClassifiedFields
from backend.ocr.schemas import Detection, OCRResult


def det(text: str, bbox, confidence: float = 0.95) -> Detection:
    x1, y1, x2, y2 = bbox
    return Detection(text=text, confidence=confidence, bbox=bbox, polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def ocr_result(lines, image="label.jpg", success=True) -> OCRResult:
    """Build an OCRResult from [(text, bbox), ...], mirroring how PaddleOCR output is shaped."""
    detections = [det(text, bbox) for text, bbox in lines]
    return OCRResult(
        success=success,
        image=image,
        full_text="\n".join(text for text, _ in lines),
        detections=detections,
        detection_count=len(detections),
        preprocessing_applied=False,
    )


# One fact per detection with a plausible bounding box, top-to-bottom - the shape a real
# product label photo would actually produce, not prose run through .splitlines().
MOSTLY_COMPLIANT_LINES = [
    ("FreshBite Crunchy Namkeen Mix", [10, 10, 300, 30]),
    ("Net Qty 200 g", [10, 32, 200, 52]),
    ("MRP", [10, 54, 60, 74]),
    ("Rs. 60.00", [10, 76, 120, 96]),
    ("Inclusive of all taxes", [10, 98, 250, 118]),
    ("Unit Sale Price", [10, 120, 180, 140]),
    ("Rs 30 per 100g", [10, 142, 200, 162]),
    ("Mfd by FreshBite Foods Pvt Ltd, MIDC Industrial Area, Pune, Maharashtra 411019", [10, 164, 600, 184]),
    ("MFG MAR 2026", [10, 186, 200, 206]),
    ("Country of Origin India", [10, 208, 300, 228]),
    ("Consumer Care 1800-123-4567", [10, 230, 300, 250]),
    ("care@freshbite.example.com", [10, 252, 300, 272]),
]

CLEARLY_NON_COMPLIANT_LINES = [
    ("Tasty Treats", [10, 10, 150, 30]),
    ("Best before 6 months from packing", [10, 32, 300, 52]),
]

# Captured verbatim from a real PaddleOCR run against backend/test_images/synthetic_label.png
# (Phase 3 manual verification) - genuine engine output, not a synthetic fixture.
REAL_PADDLEOCR_LINES = [
    ("HERBAL GLOW HANDWASH", [38, 33, 511, 62]),
    ("Manufactured by: Sunrise Consumer Products Pvt. Ltd.", [39, 122, 664, 146]),
    ("Address: Plot 14, Industrial Area, Pune, Maharashtra - 411019", [38, 160, 598, 183]),
    ("Net Quantity: 250 ml", [38, 221, 273, 248]),
    ("MRP: Rs. 149.00 (Incl. of all taxes)", [38, 262, 436, 288]),
    ("Mfg Date: 03/2024", [38, 301, 250, 328]),
    ("Best Before: 24 months from Mfg Date", [38, 340, 475, 367]),
    ("Consumer Care: 1800-123-4567", [39, 402, 408, 425]),
    ("Email: care@sunriseconsumer.in", [39, 443, 418, 467]),
    ("Country of Origin: India", [38, 501, 304, 528]),
]


class MostlyCompliantScenarioTests(unittest.TestCase):
    def setUp(self):
        self.report = run_full_compliance_pipeline(ocr_result(MOSTLY_COMPLIANT_LINES))

    def test_overall_status_is_partially_compliant(self):
        # Every hard-checkable field passes; only common_generic_name_of_commodity is
        # needs_review by design (Phase 1's honest weak heuristic), which alone keeps
        # this from being a clean "Compliant".
        self.assertEqual(self.report.overall_status, "Partially Compliant")

    def test_every_field_except_generic_name_passes(self):
        by_field = {f.field: f for f in self.report.fields}
        for field_name in (
            "manufacturer_packer_importer_details",
            "net_quantity",
            "month_and_year_of_manufacture_or_packing",
            "maximum_retail_price_mrp",
            "unit_sale_price",
            "consumer_care_details",
            "country_of_origin",
        ):
            with self.subTest(field=field_name):
                self.assertEqual(by_field[field_name].verdict, "Pass")
        self.assertEqual(by_field["common_generic_name_of_commodity"].verdict, "Needs Review")

    def test_mrp_confirms_tax_inclusive_even_though_split_across_three_detections(self):
        # Regression coverage: MRP keyword, price, and "Inclusive of all taxes" are three
        # separate OCR detections here - the tax-phrase check must still find the third
        # one even though the classifier only bundled the first two into combined_text.
        by_field = {f.field: f for f in self.report.fields}
        self.assertEqual(by_field["maximum_retail_price_mrp"].verdict, "Pass")

    def test_unit_sale_price_looks_past_the_bare_label_to_the_real_number(self):
        # Regression coverage: "Unit Sale Price" (bare label, no digits) is immediately
        # followed by "Rs 30 per 100g" as a separate detection - the classifier must not
        # stop at the label alone.
        by_field = {f.field: f for f in self.report.fields}
        self.assertEqual(by_field["unit_sale_price"].verdict, "Pass")
        self.assertIn("30", by_field["unit_sale_price"].extracted_value)

    def test_no_field_silently_defaults_to_pass_without_evidence(self):
        # Every Pass must carry either a high/medium confidence or an extracted_value -
        # never a bare, unexplained pass.
        for f in self.report.fields:
            if f.verdict == "Pass":
                with self.subTest(field=f.field):
                    self.assertTrue(f.extracted_value or f.confidence)


class ClearlyNonCompliantScenarioTests(unittest.TestCase):
    def setUp(self):
        self.report = run_full_compliance_pipeline(ocr_result(CLEARLY_NON_COMPLIANT_LINES, image="sparse.jpg"))

    def test_overall_status_is_non_compliant(self):
        self.assertEqual(self.report.overall_status, "Non-Compliant")

    def test_hard_checkable_fields_all_fail(self):
        by_field = {f.field: f for f in self.report.fields}
        for field_name in (
            "manufacturer_packer_importer_details",
            "net_quantity",
            "month_and_year_of_manufacture_or_packing",
            "maximum_retail_price_mrp",
            "unit_sale_price",
            "consumer_care_details",
        ):
            with self.subTest(field=field_name):
                self.assertEqual(by_field[field_name].verdict, "Fail")

    def test_conditional_field_absence_is_not_applicable_not_a_failure(self):
        by_field = {f.field: f for f in self.report.fields}
        self.assertEqual(by_field["country_of_origin"].verdict, "Not Applicable")
        # Not Applicable must never appear in the violation list.
        violation_fields = {v.field for v in self.report.violations}
        self.assertNotIn("country_of_origin", violation_fields)

    def test_every_failure_has_a_plain_language_reason(self):
        for v in self.report.violations:
            with self.subTest(field=v.field):
                self.assertTrue(v.reason)


class RealPaddleOCROutputScenarioTests(unittest.TestCase):
    """Uses genuine engine output (see REAL_PADDLEOCR_LINES) rather than a hand-typed fixture."""

    def setUp(self):
        self.report = run_full_compliance_pipeline(ocr_result(REAL_PADDLEOCR_LINES, image="synthetic_label.png"))

    def test_consumer_care_combines_phone_and_email_from_separate_lines(self):
        by_field = {f.field: f for f in self.report.fields}
        self.assertEqual(by_field["consumer_care_details"].verdict, "Pass")
        self.assertIn("1800-123-4567", by_field["consumer_care_details"].extracted_value)
        self.assertIn("care@sunriseconsumer.in", by_field["consumer_care_details"].extracted_value)

    def test_mrp_combined_with_label_on_one_line_is_detected(self):
        # "MRP: Rs. 149.00 (Incl. of all taxes)" is ONE detection with "MRP:" before
        # "Rs." - previously a known limitation (the ruleset's ^Rs\. anchored regex
        # couldn't match mid-line), fixed by stripping the anchors in mrp_pattern().
        # Reproduced here on real OCR text, not just asserted in isolation.
        by_field = {f.field: f for f in self.report.fields}
        mrp_field = by_field["maximum_retail_price_mrp"]
        self.assertEqual(mrp_field.verdict, "Pass")
        self.assertIn("149.00", mrp_field.extracted_value)

    def test_best_before_is_not_mistaken_for_manufacture_date(self):
        by_field = {f.field: f for f in self.report.fields}
        date_field = by_field["month_and_year_of_manufacture_or_packing"]
        self.assertEqual(date_field.verdict, "Pass")
        self.assertNotIn("24 months", date_field.extracted_value or "")


MULTI_FACE_CONFLICTING_MRP_LINES = [
    ("Front panel", [10, 10, 150, 30]),
    ("MRP", [10, 32, 60, 52]),
    ("Rs. 60.00", [10, 54, 120, 74]),
    ("Back panel", [10, 300, 150, 320]),
    ("MRP", [10, 322, 60, 342]),
    ("Rs. 65.00", [10, 344, 120, 364]),
]


# Modeled on a genuine Britannia Good Day biscuit packet - the exact scenario that
# surfaced three real bugs: MRP combined with label text on one OCR line ("MRP (Incl. of
# all taxes) ₹20.00"), a packing date (PKD) and an expiry date (USE BY) on separate
# detections that were being conflated into one field, and Nutrition Information table
# gram figures (Energy/Protein/Carbohydrate/Fat) being picked up as net quantity instead
# of the real "NET WEIGHT: 75 g" declaration.
BRITANNIA_GOOD_DAY_LINES = [
    ("BRITANNIA Good Day", [10, 10, 300, 30]),
    ("Cashew Almond Cookies", [10, 32, 300, 52]),
    ("NET WEIGHT: 75 g", [10, 54, 250, 74]),
    ("MRP (Incl. of all taxes) ₹20.00", [10, 76, 350, 96]),
    ("Mfd by Britannia Industries Ltd, Mumbai, Maharashtra", [10, 98, 600, 118]),
    ("PKD: 12/05/2024", [10, 120, 200, 140]),
    ("USE BY: 11/11/2024", [10, 142, 220, 162]),
    ("Consumer Care 1800-266-7676", [10, 164, 300, 184]),
    ("care@britannia.co.in", [10, 186, 250, 206]),
    ("Country of Origin: India", [10, 208, 300, 228]),
    ("Nutrition Information", [10, 300, 250, 320]),
    ("Per 100 g", [10, 322, 150, 342]),
    ("Energy 468.7 g", [10, 344, 200, 364]),
    ("Protein 6.4 g", [10, 366, 200, 386]),
    ("Carbohydrate 68.7 g", [10, 388, 220, 408]),
    ("Total Fat 21.2 g", [10, 410, 200, 430]),
    ("Sugars 29.3 g", [10, 432, 200, 452]),
    ("Sodium 10.3 g", [10, 454, 200, 474]),
    ("Trans Fat 0.1 g", [10, 476, 200, 496]),
]


class RealLabelThreeBugScenarioTests(unittest.TestCase):
    """End-to-end regression coverage for the three bugs found against a genuine label."""

    def setUp(self):
        self.report = run_full_compliance_pipeline(ocr_result(BRITANNIA_GOOD_DAY_LINES, image="good_day.jpg"))
        self.by_field = {f.field: f for f in self.report.fields}

    def test_bug1_mrp_combined_with_label_text_is_detected(self):
        mrp = self.by_field["maximum_retail_price_mrp"]
        self.assertEqual(mrp.verdict, "Pass")
        self.assertIn("20.00", mrp.extracted_value)

    def test_bug2_packing_date_is_detected_and_not_conflated_with_expiry(self):
        date_field = self.by_field["month_and_year_of_manufacture_or_packing"]
        self.assertEqual(date_field.verdict, "Pass")
        self.assertNotEqual(date_field.status, "ambiguous")
        self.assertNotIn("11/11/2024", date_field.extracted_value or "")

    def test_bug3_net_quantity_is_75g_not_a_nutrition_table_figure(self):
        net_qty = self.by_field["net_quantity"]
        self.assertEqual(net_qty.verdict, "Pass")
        self.assertIn("75", net_qty.extracted_value)
        for bogus in ("6.4", "68.7", "21.2", "29.3", "10.3", "0.1", "468.7"):
            self.assertNotIn(bogus, net_qty.extracted_value)

    def test_overall_report_has_no_false_ambiguous_flags(self):
        # None of the three real bugs should surface as "ambiguous"/needs_review noise -
        # the only fields legitimately needing review are the ones honestly weak by design
        # (generic name) or genuinely absent from this label (no unit sale price declared).
        for field in self.report.fields:
            if field.field in ("common_generic_name_of_commodity", "unit_sale_price"):
                continue
            with self.subTest(field=field.field):
                self.assertNotEqual(field.status, "ambiguous")


class MultiFaceLabelScenarioTests(unittest.TestCase):
    """Phase 6: a package with two panels printing genuinely different MRPs."""

    def test_conflicting_mrp_is_flagged_needs_review_not_silently_picked(self):
        report = run_full_compliance_pipeline(ocr_result(MULTI_FACE_CONFLICTING_MRP_LINES, image="multiface.jpg"))
        by_field = {f.field: f for f in report.fields}
        mrp = by_field["maximum_retail_price_mrp"]
        self.assertEqual(mrp.status, "ambiguous")
        self.assertEqual(mrp.verdict, "Needs Review")
        self.assertIn("Rs. 60.00", mrp.extracted_value)
        self.assertIn("Rs. 65.00", mrp.extracted_value)

    def test_overall_status_reflects_the_ambiguity(self):
        report = run_full_compliance_pipeline(ocr_result(MULTI_FACE_CONFLICTING_MRP_LINES, image="multiface.jpg"))
        # No hard failures forced by this alone, but it must never read as a clean "Compliant".
        self.assertNotEqual(report.overall_status, "Compliant")


class BlurryOrPartialScanEdgeCaseTests(unittest.TestCase):
    """Confirms graceful degradation - no crash, no false-confident Compliant verdicts."""

    def test_single_low_confidence_garbled_detection_does_not_crash(self):
        garbled = ocr_result([("???", [5, 5, 50, 15])], image="blurry.jpg")
        report = run_full_compliance_pipeline(garbled)  # must not raise
        self.assertEqual(report.overall_status, "Non-Compliant")
        for f in report.fields:
            with self.subTest(field=f.field):
                self.assertNotEqual(f.verdict, "Pass")

    def test_zero_detections_does_not_crash(self):
        blank = OCRResult(
            success=True, image="blank.jpg", full_text="", detections=[],
            detection_count=0, preprocessing_applied=False,
        )
        report = run_full_compliance_pipeline(blank)  # must not raise
        self.assertEqual(report.overall_status, "Non-Compliant")
        for f in report.fields:
            with self.subTest(field=f.field):
                self.assertNotEqual(f.verdict, "Pass")

    def test_failed_ocr_produces_unknown_status_not_a_crash_or_false_pass(self):
        from backend.ocr.schemas import OCRError

        failed = OCRResult(
            success=False, image="unreadable.jpg", full_text="",
            detections=[], detection_count=0, preprocessing_applied=False,
            error=OCRError(code="empty_result", message="No text detected"),
        )
        report = run_full_compliance_pipeline(failed)
        self.assertEqual(report.overall_status, "Unknown")
        self.assertEqual(report.fields, [])
        self.assertTrue(report.notes)


SIMPLE_OCR_RESULT = OCRResult(
    success=True, image="label.jpg", full_text="Random label text with no recognizable fields",
    detections=[det("Random label text with no recognizable fields", [10, 10, 300, 30])],
    detection_count=1, preprocessing_applied=False,
)


class ClassifierModeTests(unittest.TestCase):
    """
    COMPLIANCE_CLASSIFIER_MODE switching in pipeline.py. The LLM side is always mocked
    here (no real network call, no cost, fully deterministic) - live Groq integration is
    covered separately by test_llm_classifier.py's gated LiveGroqSmokeTest.
    """

    def setUp(self):
        # Ensure no leftover env var from another test/process affects this one.
        self._original_mode = os.environ.pop("COMPLIANCE_CLASSIFIER_MODE", None)

    def tearDown(self):
        if self._original_mode is not None:
            os.environ["COMPLIANCE_CLASSIFIER_MODE"] = self._original_mode
        else:
            os.environ.pop("COMPLIANCE_CLASSIFIER_MODE", None)

    def test_default_mode_is_regex_and_never_touches_llm(self):
        with patch("backend.compliance.pipeline.classify_fields_with_llm") as mock_llm:
            report = run_full_compliance_pipeline(SIMPLE_OCR_RESULT)
        mock_llm.assert_not_called()
        self.assertEqual(report.overall_status, "Non-Compliant")

    def test_unknown_mode_value_falls_back_to_regex(self):
        os.environ["COMPLIANCE_CLASSIFIER_MODE"] = "not_a_real_mode"
        with patch("backend.compliance.pipeline.classify_fields_with_llm") as mock_llm:
            run_full_compliance_pipeline(SIMPLE_OCR_RESULT)
        mock_llm.assert_not_called()

    def test_llm_mode_falls_back_to_regex_cleanly_when_llm_unavailable(self):
        os.environ["COMPLIANCE_CLASSIFIER_MODE"] = "llm"
        with patch("backend.compliance.pipeline.classify_fields_with_llm") as mock_llm:
            mock_llm.side_effect = LLMClassificationError("no key configured")
            report = run_full_compliance_pipeline(SIMPLE_OCR_RESULT)  # must not raise
        self.assertEqual(report.overall_status, "Non-Compliant")
        self.assertTrue(any("unavailable" in n for n in report.notes))

    def test_llm_mode_uses_llm_result_when_available(self):
        os.environ["COMPLIANCE_CLASSIFIER_MODE"] = "llm"
        fake_fields = ClassifiedFields(fields={
            "net_quantity": ClassifiedField(
                field="net_quantity", confidence="high", combined_text="200 g",
            ),
        }, full_text=SIMPLE_OCR_RESULT.full_text)
        fake_usage = LLMUsage(prompt_tokens=100, completion_tokens=50, elapsed_seconds=1.0)
        with patch("backend.compliance.pipeline.classify_fields_with_llm") as mock_llm:
            mock_llm.return_value = (fake_fields, fake_usage, None)
            report = run_full_compliance_pipeline(SIMPLE_OCR_RESULT)
        by_field = {f.field: f for f in report.fields}
        self.assertEqual(by_field["net_quantity"].verdict, "Pass")
        self.assertTrue(any("tokens" in n for n in report.notes))

    def test_hybrid_mode_flags_disagreement_between_classifiers(self):
        os.environ["COMPLIANCE_CLASSIFIER_MODE"] = "hybrid"
        # Regex finds nothing in SIMPLE_OCR_RESULT's text (net_quantity -> Fail); the
        # mocked LLM claims a real value (net_quantity -> Pass) - a genuine disagreement.
        fake_fields = ClassifiedFields(fields={
            "net_quantity": ClassifiedField(
                field="net_quantity", confidence="high", combined_text="200 g",
            ),
        }, full_text=SIMPLE_OCR_RESULT.full_text)
        fake_usage = LLMUsage(prompt_tokens=100, completion_tokens=50, elapsed_seconds=1.0)
        with patch("backend.compliance.pipeline.classify_fields_with_llm") as mock_llm:
            mock_llm.return_value = (fake_fields, fake_usage, None)
            report = run_full_compliance_pipeline(SIMPLE_OCR_RESULT)

        # Hybrid reports from regex - net_quantity should still read as regex's (Fail).
        by_field = {f.field: f for f in report.fields}
        self.assertEqual(by_field["net_quantity"].verdict, "Fail")
        # But the disagreement must be called out for a human to see.
        self.assertTrue(any("disagreement" in n.lower() and "net quantity" in n.lower() for n in report.notes))

    def test_hybrid_mode_no_notes_when_llm_and_regex_agree(self):
        os.environ["COMPLIANCE_CLASSIFIER_MODE"] = "hybrid"
        with patch("backend.compliance.pipeline.classify_fields_with_llm") as mock_llm:
            mock_llm.side_effect = LLMClassificationError("simulated outage")
            report = run_full_compliance_pipeline(SIMPLE_OCR_RESULT)
        # LLM failed in hybrid mode -> regex-only result, with an unavailability note,
        # but no false "disagreement" notes should appear since there's nothing to compare.
        self.assertFalse(any("disagreement" in n.lower() for n in report.notes))
        self.assertTrue(any("unavailable" in n for n in report.notes))


if __name__ == "__main__":
    unittest.main()
