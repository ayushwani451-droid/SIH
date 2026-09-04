"""
Unit tests for backend/compliance/field_classifier.py (Phase 2).

Run with:
    backend\\.venv\\Scripts\\python.exe -m unittest backend.compliance.test_field_classifier -v
"""
from __future__ import annotations

import unittest

from backend.compliance.field_classifier import DetectionInput, classify_fields
from backend.compliance.rule_engine import check_manufacture_date, check_mrp, check_net_quantity


def det(text: str, bbox=None, confidence: float = 0.95) -> DetectionInput:
    return DetectionInput(text=text, confidence=confidence, bbox=bbox or [])


class ClassifyFieldsTests(unittest.TestCase):
    def test_keyword_and_value_in_same_detection_is_high_confidence(self):
        detections = [det("Net Qty 200 g", bbox=[10, 10, 200, 30])]
        result = classify_fields("Net Qty 200 g", detections)
        field = result.get("net_quantity")
        self.assertEqual(field.confidence, "high")
        self.assertEqual(len(field.matched_detections), 1)

    def test_keyword_and_value_in_adjacent_detections_is_medium_confidence(self):
        detections = [
            det("MRP", bbox=[10, 10, 60, 30]),
            det("Rs. 60.00", bbox=[10, 32, 90, 52]),
        ]
        result = classify_fields("MRP\nRs. 60.00", detections)
        field = result.get("maximum_retail_price_mrp")
        self.assertEqual(field.confidence, "medium")
        self.assertEqual(len(field.matched_detections), 2)
        self.assertIn("Rs. 60.00", field.combined_text)

    def test_bare_value_with_no_keyword_anywhere_is_low_confidence(self):
        # No "MRP" label detected anywhere, just a price-shaped line.
        detections = [det("Rs. 60.00", bbox=[10, 10, 90, 30])]
        result = classify_fields("Rs. 60.00", detections)
        field = result.get("maximum_retail_price_mrp")
        self.assertEqual(field.confidence, "low")

    def test_keyword_with_no_nearby_value_is_low_confidence(self):
        detections = [
            det("MRP", bbox=[10, 10, 40, 30]),
            det("Some unrelated line", bbox=[10, 32, 200, 52]),
            det("Another unrelated line", bbox=[10, 54, 200, 74]),
            det("Yet another unrelated line", bbox=[10, 76, 200, 96]),
        ]
        result = classify_fields("MRP\nSome unrelated line\nAnother unrelated line\nYet another unrelated line", detections)
        field = result.get("maximum_retail_price_mrp")
        self.assertEqual(field.confidence, "low")
        self.assertEqual(field.anchor_keyword, "mrp")

    def test_fused_field_manufacturer_details_is_high_confidence_on_single_hit(self):
        detections = [det("Mfd by Acme Foods, Delhi", bbox=[10, 10, 300, 30])]
        result = classify_fields("Mfd by Acme Foods, Delhi", detections)
        field = result.get("manufacturer_packer_importer_details")
        self.assertEqual(field.confidence, "high")

    def test_expiry_date_is_not_misclassified_as_manufacture_date(self):
        detections = [
            det("MFG MAR 2026", bbox=[10, 10, 150, 30]),
            det("Best Before DEC 2026", bbox=[10, 32, 200, 52]),
        ]
        result = classify_fields("MFG MAR 2026\nBest Before DEC 2026", detections)
        field = result.get("month_and_year_of_manufacture_or_packing")
        # Should latch onto the MFG line, not the Best Before line.
        self.assertIn("MAR 2026", field.combined_text)
        self.assertNotIn("DEC 2026", field.combined_text)

    def test_no_matches_returns_empty_classified_field(self):
        detections = [det("Crunchy and delicious!", bbox=[10, 10, 200, 30])]
        result = classify_fields("Crunchy and delicious!", detections)
        field = result.get("net_quantity")
        self.assertIsNone(field.confidence)
        self.assertEqual(field.matched_detections, [])

    def test_common_generic_name_is_never_classified(self):
        detections = [det("FreshBite Namkeen Mix", bbox=[10, 10, 200, 30])]
        result = classify_fields("FreshBite Namkeen Mix", detections)
        field = result.get("common_generic_name_of_commodity")
        self.assertIsNone(field.confidence)

    def test_consumer_care_gathers_phone_and_email_across_separate_detections(self):
        # Regression test: caught against real PaddleOCR output where phone and email
        # land on two separate lines, neither mentioning the other. The naive
        # keyword+value classifier used to stop at the phone-only line and silently
        # drop the email, which was WORSE than not classifying at all.
        detections = [
            det("Consumer Care: 1800-123-4567", bbox=[10, 10, 300, 30]),
            det("Email: care@brand.example.com", bbox=[10, 32, 300, 52]),
        ]
        result = classify_fields(
            "Consumer Care: 1800-123-4567\nEmail: care@brand.example.com", detections
        )
        field = result.get("consumer_care_details")
        self.assertEqual(field.confidence, "high")
        self.assertIn("1800-123-4567", field.combined_text)
        self.assertIn("care@brand.example.com", field.combined_text)

    def test_accepts_plain_dict_detections(self):
        detections = [{"text": "Net Qty 200 g", "confidence": 0.9, "bbox": [1, 2, 3, 4]}]
        result = classify_fields("Net Qty 200 g", detections)
        self.assertEqual(result.get("net_quantity").confidence, "high")


class ConflictDetectionTests(unittest.TestCase):
    """Phase 6: multi-face labels with genuinely different values for the same field."""

    def test_two_different_mrps_are_flagged_ambiguous_with_both_candidates(self):
        detections = [
            det("MRP", bbox=[10, 10, 60, 30]), det("Rs. 60.00", bbox=[10, 32, 120, 52]),
            det("MRP", bbox=[10, 300, 60, 320]), det("Rs. 65.00", bbox=[10, 322, 120, 342]),
        ]
        result = classify_fields("MRP\nRs. 60.00\nMRP\nRs. 65.00", detections)
        field = result.get("maximum_retail_price_mrp")
        self.assertTrue(field.ambiguous)
        self.assertIn("Rs. 60.00", field.candidate_values)
        self.assertIn("Rs. 65.00", field.candidate_values)

    def test_same_mrp_repeated_twice_is_not_flagged_ambiguous(self):
        detections = [
            det("MRP", bbox=[10, 10, 60, 30]), det("Rs. 60.00", bbox=[10, 32, 120, 52]),
            det("MRP", bbox=[10, 300, 60, 320]), det("Rs. 60.00", bbox=[10, 322, 120, 342]),
        ]
        result = classify_fields("MRP\nRs. 60.00\nMRP\nRs. 60.00", detections)
        field = result.get("maximum_retail_price_mrp")
        self.assertFalse(field.ambiguous)

    def test_two_different_net_quantities_are_flagged_ambiguous(self):
        detections = [det("Net Qty 200 g", bbox=[10, 10, 200, 30]), det("Net Qty 250 g", bbox=[10, 32, 200, 52])]
        result = classify_fields("Net Qty 200 g\nNet Qty 250 g", detections)
        field = result.get("net_quantity")
        self.assertTrue(field.ambiguous)

    def test_unit_price_detection_does_not_falsely_conflict_with_net_quantity(self):
        # Regression: "Rs 30 per 100g" contains "100g", which the net_quantity regex also
        # matches - it must not be mistaken for a second, conflicting net quantity.
        detections = [
            det("Net Qty 200 g", bbox=[10, 10, 200, 30]),
            det("Unit Sale Price", bbox=[10, 32, 180, 52]),
            det("Rs 30 per 100g", bbox=[10, 54, 200, 74]),
        ]
        result = classify_fields("Net Qty 200 g\nUnit Sale Price\nRs 30 per 100g", detections)
        field = result.get("net_quantity")
        self.assertFalse(field.ambiguous)
        self.assertEqual(field.confidence, "high")

    def test_manufacturer_and_country_are_never_conflict_checked(self):
        # Out of scope by design: free-text name comparison isn't reliable via regex.
        detections = [
            det("Mfd by Acme Foods, Delhi", bbox=[10, 10, 300, 30]),
            det("Mfd by Acme Foods Pvt Ltd, Mumbai", bbox=[10, 32, 300, 52]),
        ]
        result = classify_fields("Mfd by Acme Foods, Delhi\nMfd by Acme Foods Pvt Ltd, Mumbai", detections)
        field = result.get("manufacturer_packer_importer_details")
        self.assertFalse(field.ambiguous)


class RealLabelBugRegressionTests(unittest.TestCase):
    """
    Regression tests for three specific bugs found testing against a genuine product
    label (Britannia Good Day biscuits): MRP combined with label text on one OCR line,
    packing/expiry dates on separate detections getting conflated into one field, and
    Nutrition Information table figures being mistaken for net quantity.
    """

    def test_pkd_and_use_by_dates_on_separate_detections_are_not_conflated(self):
        # Realistic split: "PKD"/"USE BY" and their dates often land as four separate
        # OCR detections, not two combined lines. A manufacturer line with "Mfd by" sits
        # in between too, since that's a real source of keyword collision (see below).
        detections = [
            det("Mfd by Britannia Industries Ltd, Mumbai", bbox=[10, 10, 400, 30]),
            det("PKD: 12/05/2024", bbox=[10, 32, 200, 52]),
            det("USE BY: 11/11/2024", bbox=[10, 54, 220, 74]),
        ]
        full_text = "\n".join(d.text for d in detections)
        result = classify_fields(full_text, detections)
        field = result.get("month_and_year_of_manufacture_or_packing")
        self.assertFalse(field.ambiguous, "PKD and USE BY must not be flagged as conflicting values of one field")
        self.assertIn("2024", field.combined_text)
        self.assertNotIn("11/11/2024", field.combined_text)
        self.assertNotIn("USE BY", field.combined_text.upper())

    def test_manufacturer_line_does_not_falsely_anchor_the_date_field(self):
        # "Mfd by ..." contains the substring "mfd", which is also a legitimate date
        # keyword ("MFD: 12/2024") - it must not be mistaken for a date-field anchor.
        detections = [
            det("Mfd by Britannia Industries Ltd, Mumbai", bbox=[10, 10, 400, 30]),
            det("PKD: 12/05/2024", bbox=[10, 32, 200, 52]),
        ]
        full_text = "\n".join(d.text for d in detections)
        result = classify_fields(full_text, detections)
        field = result.get("month_and_year_of_manufacture_or_packing")
        # Anchor must come from the PKD line's own text, not from "Mfd by Britannia...".
        self.assertIn("PKD", field.combined_text)
        self.assertNotIn("Britannia", field.combined_text)

    def test_net_quantity_ignores_nutrition_table_even_with_no_other_gram_values(self):
        detections = [
            det("Nutrition Information", bbox=[10, 300, 250, 320]),
            det("Per 100 g", bbox=[10, 322, 150, 342]),
            det("Energy 468.7 g", bbox=[10, 344, 200, 364]),
            det("Protein 6.4 g", bbox=[10, 366, 200, 386]),
            det("Carbohydrate 68.7 g", bbox=[10, 388, 220, 408]),
            det("Total Fat 21.2 g", bbox=[10, 410, 200, 430]),
        ]
        full_text = "\n".join(d.text for d in detections)
        result = classify_fields(full_text, detections)
        field = result.get("net_quantity")
        # No real net-quantity keyword ("Net Wt" etc.) anywhere on this label at all - the
        # engine must not grab a nutrition-table figure as a fallback guess.
        self.assertIsNone(field.confidence)
        self.assertEqual(field.matched_detections, [])

    def test_net_quantity_finds_net_weight_and_ignores_nutrition_table_alongside_it(self):
        detections = [
            det("NET WEIGHT: 75 g", bbox=[10, 54, 250, 74]),
            det("Nutrition Information", bbox=[10, 300, 250, 320]),
            det("Per 100 g", bbox=[10, 322, 150, 342]),
            det("Energy 468.7 g", bbox=[10, 344, 200, 364]),
            det("Protein 6.4 g", bbox=[10, 366, 200, 386]),
            det("Carbohydrate 68.7 g", bbox=[10, 388, 220, 408]),
            det("Total Fat 21.2 g", bbox=[10, 410, 200, 430]),
            det("Sugars 29.3 g", bbox=[10, 432, 200, 452]),
            det("Sodium 10.3 g", bbox=[10, 454, 200, 474]),
            det("Trans Fat 0.1 g", bbox=[10, 476, 200, 496]),
        ]
        full_text = "\n".join(d.text for d in detections)
        result = classify_fields(full_text, detections)
        field = result.get("net_quantity")
        self.assertEqual(field.confidence, "high")
        self.assertIn("75", field.combined_text)
        self.assertFalse(field.ambiguous, "nutrition-table figures must not be treated as conflicting net quantities")


class ValidatorConsumesClassifiedFieldTests(unittest.TestCase):
    """Confirms rule_engine's Phase-2 `classified` param actually narrows the scan, using
    real classify_fields() output rather than hand-built ClassifiedField objects."""

    def test_mrp_validator_prefers_narrowed_classified_text(self):
        detections = [det("MRP"), det("Rs. 60.00"), det("Inclusive of all taxes")]
        classified = classify_fields("MRP\nRs. 60.00\nInclusive of all taxes", detections)
        result = check_mrp("MRP\nRs. 60.00\nInclusive of all taxes", classified=classified.get("maximum_retail_price_mrp"))
        self.assertEqual(result.status, "present")

    def test_net_quantity_validator_uses_classified_when_given(self):
        classified = classify_fields("Net Qty 200 g", [det("Net Qty 200 g")])
        result = check_net_quantity("irrelevant fallback text", classified=classified.get("net_quantity"))
        self.assertTrue(result.compliant)
        self.assertIn("200", result.extracted_value)

    def test_manufacture_date_validator_falls_back_when_classified_field_empty(self):
        # classified has no match at all (empty combined_text) -> falls back to scanning `text`.
        classified = classify_fields("no date info", [det("no date info")])
        result = check_manufacture_date("MFG MAR 2026", classified=classified.get("month_and_year_of_manufacture_or_packing"))
        self.assertTrue(result.compliant)


if __name__ == "__main__":
    unittest.main()
