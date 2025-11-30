from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List

from backend.app.services.template_service import (
    get_all_templates,
    find_templates,
    create_template_from_upload,
    save_template,
    fill_template,
    get_template_by_id
)
from backend.app.utils.schemas import TemplateOut, TemplateIn
from backend.app.models.models import FillTemplateRequest, DraftRequest
from backend.app.utils.dependencies import get_org_id

router = APIRouter(tags=["templates"])


# ---------------------------------------------------------
# CREATE TEMPLATE FROM UPLOAD  (ORG SAFE)
# ---------------------------------------------------------
@router.post("/create-template-from-upload/")
async def create_template_from_upload_endpoint(
    file: UploadFile = File(...),
    org_id: str = Depends(get_org_id)
):
    return await create_template_from_upload(file, org_id)


# ---------------------------------------------------------
# SAVE TEMPLATE (ORG SAFE)
# ---------------------------------------------------------
@router.post("/save-template/")
async def save_template_endpoint(
    template_data: TemplateIn,
    org_id: str = Depends(get_org_id)
):
    return await save_template(template_data, org_id)


# ---------------------------------------------------------
# GET ALL TEMPLATES (ORG SAFE)
# ---------------------------------------------------------
@router.get("/templates/", response_model=List[TemplateOut])
async def get_all_templates_endpoint(
    org_id: str = Depends(get_org_id)
):
    return await get_all_templates(org_id)


# ---------------------------------------------------------
# GET SINGLE TEMPLATE (ORG SAFE)
# ---------------------------------------------------------
@router.get("/templates/{template_id}", response_model=TemplateOut)
async def get_template_by_id_endpoint(
    template_id: str,
    org_id: str = Depends(get_org_id)
):
    return await get_template_by_id(template_id, org_id)


# ---------------------------------------------------------
# FIND / BOOTSTRAP TEMPLATES (ORG SAFE)
# ---------------------------------------------------------
@router.post("/find-templates")
async def find_templates_endpoint(
    request: DraftRequest,
    org_id: str = Depends(get_org_id)
):
    return await find_templates(request, org_id)


# ---------------------------------------------------------
# FILL TEMPLATE (ORG SAFE)
# ---------------------------------------------------------
@router.post("/fill-template")
async def fill_template_endpoint(
    request: FillTemplateRequest,
    org_id: str = Depends(get_org_id)
):
    return await fill_template(request, org_id)
