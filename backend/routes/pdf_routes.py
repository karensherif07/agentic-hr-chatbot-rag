"""
pdf_routes.py — Serves rendered PDF page images for the "Source Evidence"
panel, using the exact same render_page_to_image() from setup.py that the
original Streamlit chat_ui.py called via st.image(). No logic changed —
just exposed over HTTP instead of embedded directly in a Streamlit script.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response

from deps import get_current_employee
from setup import render_page_to_image

router = APIRouter(prefix="/api/policies", tags=["policies"])


@router.get("/page-image")
def page_image(
    source: str = Query(..., description="PDF file path, e.g. policies/eng_policy.pdf"),
    page: int = Query(..., ge=1),
    emp: dict = Depends(get_current_employee),
):
    try:
        img_bytes = render_page_to_image(source, page)
    except Exception as e:
        raise HTTPException(404, f"Could not render page: {e}")
    return Response(content=img_bytes, media_type="image/png")
