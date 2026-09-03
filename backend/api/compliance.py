"""
New, additive-only endpoint: GET /api/ocr/compliance/{scan_id}.

Looks up an already-persisted scan (created via the existing POST /api/scans in
backend/api/ocr.py), runs its stored OCR result through the Phase 3 compliance
pipeline, and returns the report.

Does NOT modify backend/api/ocr.py or backend/api/auth.py in any way - it only imports
the existing `official_user` dependency and reuses the `scans` collection exactly as
they already work, so ownership/auth behaves identically to GET /api/scans/{scan_id}.

There is no async OCR job queue anywhere in this codebase - OCR runs synchronously in
POST /api/ocr, so "already-completed OCR job" here means a scan persisted via POST
/api/scans, looked up the same way GET /api/scans/{scan_id} already does.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.api.auth import official_user
from backend.compliance.pipeline import run_full_compliance_pipeline
from backend.compliance.report_schema import ProductIdentification
from backend.database import get_database
from backend.ocr.schemas import OCRResult

router = APIRouter(prefix="/api/ocr", tags=["compliance"])


@router.get("/compliance/{scan_id}")
def get_scan_compliance(
    scan_id: str,
    profile: str = Query("e_commerce_product_listing"),
    user: Dict[str, Any] = Depends(official_user),
):
    """
    Run the Rule 6 compliance engine against an already-completed OCR scan.

    404 if no such scan exists for this official. 202 "pending" (not blocking, not an
    error) if the scan's OCR did not complete successfully - as create_scan() is written
    today every persisted scan already has success=True, so this branch is defensive
    for whenever that changes, not something reachable via the current storage path.
    """
    scan = get_database().scans.find_one({"id": scan_id, "userId": user["_id"]}, {"_id": 0})
    if scan is None:
        raise HTTPException(status_code=404, detail="Scan not found")

    if not scan.get("success"):
        return JSONResponse(status_code=202, content={
            "scanId": scan_id,
            "status": "pending",
            "detail": "OCR for this scan has not completed successfully yet; no compliance report is available.",
        })

    ocr_payload = scan.get("ocr")
    if not ocr_payload:
        raise HTTPException(status_code=422, detail="This scan has no stored OCR result to evaluate.")

    try:
        ocr_result = OCRResult.model_validate(ocr_payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stored OCR result is malformed: {exc}") from None

    product = ProductIdentification(
        scan_id=scan_id,
        image=scan.get("image"),
        product_name=scan.get("productName"),
        category=scan.get("category"),
        manufacturer=scan.get("manufacturer"),
    )

    try:
        report = run_full_compliance_pipeline(ocr_result, profile=profile, product=product)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    return report.model_dump(mode="json")
