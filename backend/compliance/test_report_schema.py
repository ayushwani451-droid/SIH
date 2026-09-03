"""
Unit tests for backend/compliance/report_schema.py (Phase 4).

Run with:
    backend\\.venv\\Scripts\\python.exe -m unittest backend.compliance.test_report_schema -v
"""
from __future__ import annotations

import unittest

from backend.compliance.report_schema import ProductIdentification, build_compliance_report


def _pipeline_result(fields):
    return {
        "profile": "e_commerce_product_listing",
        "ruleset_id": "IN-LM-RULES-2011",
        "ruleset_version": "2026.1.0",
        "evaluated_rules": ["LMPC-R6-MANDATORY-DECLARATIONS"],
        "fields": fields,
        "overall_status": "Non-Compliant",
        "exemptions": {"exempt": False, "applicable_exemptions": [], "reason": "n/a"},
        "notes": [],
    }


class VerdictDerivationTests(unittest.TestCase):
    def test_pass_for_compliant_true(self):
        result = _pipeline_result([{
            "field": "net_quantity", "status": "present", "extracted_value": "200 g",
            "compliant": True, "reason": None, "conditional_field": False,
        }])
        report = build_compliance_report(result)
        self.assertEqual(report.fields[0].verdict, "Pass")
        self.assertEqual(report.violations, [])

    def test_fail_for_hard_failure(self):
        result = _pipeline_result([{
            "field": "net_quantity", "status": "absent", "extracted_value": None,
            "compliant": False, "reason": "No quantity found.", "conditional_field": False,
        }])
        report = build_compliance_report(result)
        self.assertEqual(report.fields[0].verdict, "Fail")
        self.assertEqual(len(report.violations), 1)
        self.assertEqual(report.violations[0].severity, "violation")

    def test_needs_review(self):
        result = _pipeline_result([{
            "field": "common_generic_name_of_commodity", "status": "ambiguous", "extracted_value": None,
            "compliant": "needs_review", "reason": "Weak heuristic.", "conditional_field": False,
        }])
        report = build_compliance_report(result)
        self.assertEqual(report.fields[0].verdict, "Needs Review")
        self.assertEqual(report.violations[0].severity, "needs_review")

    def test_not_applicable_for_conditional_absent_field(self):
        result = _pipeline_result([{
            "field": "country_of_origin", "status": "absent", "extracted_value": None,
            "compliant": True, "reason": "Conditional field.", "conditional_field": True,
        }])
        report = build_compliance_report(result)
        self.assertEqual(report.fields[0].verdict, "Not Applicable")
        # Not Applicable must never show up as a violation, even though it has a reason.
        self.assertEqual(report.violations, [])

    def test_field_label_falls_back_to_title_case_for_unknown_field(self):
        result = _pipeline_result([{
            "field": "some_new_field", "status": "present", "extracted_value": "x",
            "compliant": True, "reason": None, "conditional_field": False,
        }])
        report = build_compliance_report(result)
        self.assertEqual(report.fields[0].field_label, "Some New Field")


class MatchedDetectionsTests(unittest.TestCase):
    def test_matched_detections_with_bbox_are_carried_through(self):
        result = _pipeline_result([{
            "field": "net_quantity", "status": "present", "extracted_value": "200 g",
            "compliant": True, "reason": None, "conditional_field": False,
            "matched_detections": [{"text": "Net Qty 200 g", "bbox": [10, 32, 200, 52], "confidence": 0.95}],
        }])
        report = build_compliance_report(result)
        detections = report.fields[0].matched_detections
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox, [10, 32, 200, 52])
        self.assertEqual(detections[0].text, "Net Qty 200 g")

    def test_missing_matched_detections_defaults_to_empty_list(self):
        result = _pipeline_result([{
            "field": "net_quantity", "status": "absent", "extracted_value": None,
            "compliant": False, "reason": "not found", "conditional_field": False,
        }])
        report = build_compliance_report(result)
        self.assertEqual(report.fields[0].matched_detections, [])


class OverallConfidenceTests(unittest.TestCase):
    def test_none_when_no_field_has_confidence(self):
        result = _pipeline_result([{
            "field": "net_quantity", "status": "present", "extracted_value": "200 g",
            "compliant": True, "reason": None, "conditional_field": False,
        }])
        report = build_compliance_report(result)
        self.assertIsNone(report.overall_confidence)

    def test_takes_the_weakest_confidence_present(self):
        result = _pipeline_result([
            {"field": "net_quantity", "status": "present", "extracted_value": "200 g",
             "compliant": True, "reason": None, "conditional_field": False, "confidence": "high"},
            {"field": "maximum_retail_price_mrp", "status": "present", "extracted_value": "Rs. 60.00",
             "compliant": True, "reason": None, "conditional_field": False, "confidence": "low"},
        ])
        report = build_compliance_report(result)
        self.assertEqual(report.overall_confidence, "low")


class ProductIdentificationTests(unittest.TestCase):
    def test_explicit_product_is_used_as_is(self):
        product = ProductIdentification(scan_id="SCN-1", image="label.png", product_name="Widget")
        result = _pipeline_result([])
        report = build_compliance_report(result, product=product)
        self.assertEqual(report.product.scan_id, "SCN-1")
        self.assertEqual(report.product.product_name, "Widget")

    def test_falls_back_to_source_image_when_no_product_given(self):
        result = _pipeline_result([])
        result["source"] = {"image": "label.png"}
        report = build_compliance_report(result)
        self.assertEqual(report.product.image, "label.png")
        self.assertIsNone(report.product.scan_id)


if __name__ == "__main__":
    unittest.main()
