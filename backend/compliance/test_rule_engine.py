"""
Unit tests for the standalone Legal Metrology rule engine.

Run with:
    backend\\.venv\\Scripts\\python.exe -m unittest backend.compliance.test_rule_engine -v

(from the Portal directory, so the `backend` package resolves).
"""
from __future__ import annotations

import unittest

from backend.compliance.rule_engine import (
    check_common_generic_name,
    check_consumer_care_details,
    check_country_of_origin,
    check_exemptions,
    check_manufacture_date,
    check_manufacturer_packer_importer_details,
    check_mrp,
    check_net_quantity,
    check_unit_sale_price,
    get_profile,
    load_ruleset,
    run_compliance_check,
)

# Written as one fact per line, mirroring how OCRResult.full_text is actually built
# (each detected text line joined with newlines) rather than as free-flowing prose.
COMPLIANT_SAMPLE = "\n".join([
    "FreshBite Crunchy Namkeen Mix",
    "Net Qty 200 g",
    "MRP",
    "Rs. 60.00",
    "Inclusive of all taxes",
    "Unit Sale Price",
    "Rs 30 per 100g",
    "Mfd by FreshBite Foods Pvt Ltd, MIDC Industrial Area, Pune, Maharashtra 411019",
    "MFG MAR 2026",
    "Country of Origin India",
    "Consumer Care 1800-123-4567",
    "care@freshbite.example.com",
])

MISSING_SAMPLE = "\n".join([
    "Tasty Treats",
    "Best before 6 months from packing",
])


class FieldValidatorTests(unittest.TestCase):
    def test_net_quantity_detects_value(self):
        result = check_net_quantity("Net Qty 200 g")
        self.assertEqual(result.status, "present")
        self.assertTrue(result.compliant)
        self.assertIn("200", result.extracted_value)

    def test_net_quantity_absent(self):
        result = check_net_quantity("No quantity info here")
        self.assertEqual(result.status, "absent")
        self.assertFalse(result.compliant)

    def test_manufacturer_details_recognizes_common_prefixes(self):
        for line in (
            "Mfd by Acme Foods, Delhi",
            "Packed by Acme Foods",
            "Marketed by Acme Traders",
            # Real-world variants (found on a genuine Parle-G label) - "manufactured FOR"
            # denotes a contract-manufacturing arrangement, still a valid Rule 6 declaration.
            "MANUFACTURED FOR Parle Biscuits Pvt. Ltd.",
            "Packed for Acme Foods, Mumbai",
            "Mfd for Acme Foods",
        ):
            with self.subTest(line=line):
                result = check_manufacturer_packer_importer_details(line)
                self.assertEqual(result.status, "present")
                self.assertTrue(result.compliant)

    def test_manufacturer_details_absent(self):
        result = check_manufacturer_packer_importer_details("Crunchy and delicious!")
        self.assertFalse(result.compliant)

    def test_generic_name_is_always_needs_review_when_text_present(self):
        result = check_common_generic_name("Some label text")
        self.assertEqual(result.compliant, "needs_review")

    def test_generic_name_absent_when_no_text(self):
        result = check_common_generic_name("")
        self.assertFalse(result.compliant)

    def test_manufacture_date_accepts_month_name_format(self):
        result = check_manufacture_date("MFG MAR 2026")
        self.assertTrue(result.compliant)
        self.assertEqual(result.extracted_value, "MAR 2026")

    def test_manufacture_date_rejects_future_date(self):
        result = check_manufacture_date("MFG DEC 2099")
        self.assertFalse(result.compliant)
        self.assertIn("future", result.reason.lower())

    def test_manufacture_date_absent(self):
        result = check_manufacture_date("Best before 6 months from packing")
        self.assertEqual(result.status, "absent")
        self.assertFalse(result.compliant)

    def test_mrp_matches_on_own_line_and_confirms_tax_inclusive(self):
        text = "MRP\nRs. 60.00\nInclusive of all taxes"
        result = check_mrp(text)
        self.assertEqual(result.status, "present")
        self.assertTrue(result.compliant)

    def test_mrp_needs_review_without_tax_phrase_nearby(self):
        text = "MRP\nRs. 45.00"
        result = check_mrp(text)
        self.assertEqual(result.status, "present")
        self.assertEqual(result.compliant, "needs_review")

    def test_mrp_matches_when_combined_with_label_on_one_line(self):
        # Regression test for the bug where the ruleset's ^Rs\.../₹...$ anchors required
        # the price to BE the entire line - real labels combine label and price on one
        # OCR-detected line ("MRP (Incl. of all taxes) Rs. 199.00"), which now matches
        # since mrp_pattern() strips the anchors before compiling.
        text = "MRP: Rs. 199.00 (inclusive of all taxes)"
        result = check_mrp(text)
        self.assertEqual(result.status, "present")
        self.assertTrue(result.compliant)
        self.assertIn("199.00", result.extracted_value)

    def test_mrp_matches_rupee_symbol_combined_with_label(self):
        text = "MRP (Incl. of all taxes) ₹20.00"
        result = check_mrp(text)
        self.assertEqual(result.status, "present")
        self.assertTrue(result.compliant)
        self.assertIn("20.00", result.extracted_value)

    def test_consumer_care_finds_phone_and_email(self):
        result = check_consumer_care_details("Consumer Care 1800-123-4567\ncare@brand.example.com")
        self.assertTrue(result.compliant)
        self.assertIn("care@brand.example.com", result.extracted_value)

    def test_consumer_care_needs_review_with_only_phone(self):
        result = check_consumer_care_details("Call us: 1800-123-4567")
        self.assertEqual(result.compliant, "needs_review")

    def test_consumer_care_absent(self):
        result = check_consumer_care_details("No contact info on this label")
        self.assertFalse(result.compliant)

    def test_country_of_origin_absent_is_not_a_hard_fail(self):
        result = check_country_of_origin("No details on this label")
        self.assertEqual(result.status, "absent")
        self.assertTrue(result.compliant)  # conditional field: absence != violation
        self.assertTrue(result.conditional_field)

    def test_country_of_origin_present(self):
        result = check_country_of_origin("Country of Origin India")
        self.assertEqual(result.status, "present")
        self.assertTrue(result.compliant)

    def test_country_of_origin_not_falsely_matched_inside_original(self):
        # Regression test: "origin" without a word boundary matched as a substring of
        # "Original" - found on a real Parle-G label ("Original Glucose Biscuits" has no
        # actual country-of-origin declaration at all, but was misread as one).
        result = check_country_of_origin("Original Glucose Biscuits")
        self.assertEqual(result.status, "absent")
        self.assertTrue(result.compliant)  # conditional field - absence is not a violation
        self.assertIsNone(result.extracted_value)

    def test_country_of_origin_bare_word_with_boundary_still_matches(self):
        # The word-boundary fix must not stop recognizing a real bare "Origin:" label.
        result = check_country_of_origin("Origin: India")
        self.assertEqual(result.status, "present")
        self.assertTrue(result.compliant)

    def test_unit_sale_price_cross_check_matches_mrp(self):
        net_qty = check_net_quantity("Net Qty 200 g")
        mrp = check_mrp("MRP\nRs. 60.00\nInclusive of all taxes")
        result = check_unit_sale_price("Rs 30 per 100g", net_quantity_result=net_qty, mrp_result=mrp)
        self.assertTrue(result.compliant)

    def test_unit_sale_price_flags_mismatch_as_needs_review_not_hard_fail(self):
        net_qty = check_net_quantity("Net Qty 200 g")
        mrp = check_mrp("MRP\nRs. 60.00\nInclusive of all taxes")
        # 30/100g over 200g implies ~60, but 999/100g implies ~1998 - way off from MRP 60.
        result = check_unit_sale_price("Rs 999 per 100g", net_quantity_result=net_qty, mrp_result=mrp)
        self.assertEqual(result.compliant, "needs_review")

    def test_unit_sale_price_absent(self):
        result = check_unit_sale_price("No pricing info at all")
        self.assertFalse(result.compliant)

    def test_exemptions_stub_reports_no_exemption(self):
        result = check_exemptions()
        self.assertFalse(result["exempt"])
        self.assertEqual(result["applicable_exemptions"], [])
        self.assertEqual(result["context_considered"], {})

    def test_exemptions_echoes_back_provided_weight_and_category(self):
        result = check_exemptions(weight_kg=0.05, category="confectionery")
        self.assertFalse(result["exempt"])  # no threshold data in the ruleset to evaluate against yet
        self.assertEqual(result["context_considered"], {"weight_kg": 0.05, "category": "confectionery"})
        self.assertIn("does not yet define any exemption rules", result["reason"])

    def test_exemptions_product_context_dict_is_merged_in(self):
        result = check_exemptions(product_context={"category": "agricultural produce", "packaging": "loose"})
        self.assertEqual(result["context_considered"]["category"], "agricultural produce")
        self.assertEqual(result["context_considered"]["packaging"], "loose")


class RulesetLoadingTests(unittest.TestCase):
    def test_load_ruleset_parses_frameworks_and_profiles(self):
        ruleset = load_ruleset()
        self.assertEqual(ruleset.ruleset_id, "IN-LM-RULES-2011")
        self.assertGreaterEqual(len(ruleset.frameworks), 2)
        self.assertIn("e_commerce_product_listing", ruleset.validation_profiles)

    def test_get_profile_raises_on_unknown_profile(self):
        ruleset = load_ruleset()
        with self.assertRaises(ValueError):
            get_profile(ruleset, "not_a_real_profile")


class RunComplianceCheckTests(unittest.TestCase):
    def test_mostly_compliant_label_is_partially_compliant(self):
        """
        Every hard-checkable field passes; overall lands on "Partially Compliant" rather
        than "Compliant" only because common_generic_name_of_commodity is, by design,
        always needs_review (we can't reliably isolate it from raw text yet) - this test
        exists to demonstrate that honesty shows up correctly in the aggregate status.
        """
        report = run_compliance_check(COMPLIANT_SAMPLE, profile="e_commerce_product_listing")
        self.assertEqual(report["overall_status"], "Partially Compliant")

        fields_by_name = {f["field"]: f for f in report["fields"]}
        for name in (
            "manufacturer_packer_importer_details",
            "net_quantity",
            "month_and_year_of_manufacture_or_packing",
            "maximum_retail_price_mrp",
            "unit_sale_price",
            "consumer_care_details",
            "country_of_origin",
        ):
            with self.subTest(field=name):
                self.assertTrue(fields_by_name[name]["compliant"] is True, fields_by_name[name])

        self.assertEqual(
            fields_by_name["common_generic_name_of_commodity"]["compliant"], "needs_review"
        )

    def test_missing_fields_label_is_non_compliant(self):
        report = run_compliance_check(MISSING_SAMPLE, profile="e_commerce_product_listing")
        self.assertEqual(report["overall_status"], "Non-Compliant")

        fields_by_name = {f["field"]: f for f in report["fields"]}
        for name in (
            "manufacturer_packer_importer_details",
            "net_quantity",
            "month_and_year_of_manufacture_or_packing",
            "maximum_retail_price_mrp",
            "unit_sale_price",
            "consumer_care_details",
        ):
            with self.subTest(field=name):
                self.assertFalse(fields_by_name[name]["compliant"], fields_by_name[name])

        # Conditional field: absent but must not be what drags the report to Non-Compliant.
        self.assertTrue(fields_by_name["country_of_origin"]["compliant"] is True)

        violation_fields = {v["field"] for v in report["violations"]}
        self.assertIn("net_quantity", violation_fields)
        self.assertIn("maximum_retail_price_mrp", violation_fields)

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            run_compliance_check(COMPLIANT_SAMPLE, profile="not_a_real_profile")

    def test_pos_hardware_audit_profile_has_no_ocr_checkable_fields(self):
        # This profile's rules (instrument verification/stamping) aren't derivable from
        # label OCR text at all - the engine should say so rather than silently pass.
        report = run_compliance_check(COMPLIANT_SAMPLE, profile="pos_hardware_audit")
        self.assertEqual(report["fields"], [])
        self.assertTrue(any("no OCR-checkable" in note for note in report["notes"]))


if __name__ == "__main__":
    unittest.main()
