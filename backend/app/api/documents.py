from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Depends
)
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict
import os
import logging

from backend.app.services.document_service import DocumentService
from backend.app.utils.dependencies import get_org_id

router = APIRouter(prefix="/documents", tags=["documents"])
document_service = DocumentService()
logger = logging.getLogger(__name__)

# -------------------------
# Request Model
# -------------------------
class UpdateFieldsRequest(BaseModel):
    insights: Dict[str, Any]


# -------------------------
# UPLOAD DOCUMENT (ORG SAFE)
# -------------------------
@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    org_id: str = Depends(get_org_id)
):
    try:
        return await document_service.upload_document(file, background_tasks, org_id)
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# GET ALL DOCUMENTS (ORG SAFE)
# -------------------------
@router.get("/")
async def get_all_documents(org_id: str = Depends(get_org_id)):
    return await document_service.get_all_documents(org_id)


# -------------------------
# GET FIELDS (ORG SAFE)
# -------------------------
@router.get("/{document_id}/fields")
async def get_document_fields(document_id: str, org_id: str = Depends(get_org_id)):
    return await document_service.get_document_fields(document_id, org_id)


# -------------------------
# UPDATE FIELDS (ORG SAFE)
# -------------------------
@router.put("/{document_id}/fields")
async def update_document_fields(
    document_id: str,
    request: UpdateFieldsRequest,
    org_id: str = Depends(get_org_id)
):
    return await document_service.update_document_fields(
        document_id, request.insights, org_id
    )


# -------------------------
# GET ORIGINAL FILE (ORG SAFE)
# -------------------------
@router.get("/{document_id}/file")
async def get_document_file(document_id: str, org_id: str = Depends(get_org_id)):
    file_path = await document_service.get_document_file_path(document_id, org_id)

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document file not found")

    ext = os.path.splitext(file_path)[1].lower()
    media_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }
    media_type = media_map.get(ext, "application/octet-stream")

    return FileResponse(file_path, media_type=media_type, filename=os.path.basename(file_path))


# -------------------------
# STATUS (ORG SAFE)
# -------------------------
@router.get("/{document_id}/status")
async def get_processing_status(document_id: str, org_id: str = Depends(get_org_id)):
    return await document_service.get_processing_status(document_id, org_id)


# -------------------------
# INSIGHTS (ORG SAFE)
# -------------------------
@router.get("/{document_id}/insights")
async def get_document_insights(document_id: str, org_id: str = Depends(get_org_id)):
    return await document_service.get_document_insights(document_id, org_id)


# -------------------------
# ASK AI (ORG SAFE)
# -------------------------
@router.post("/{document_id}/query")
async def query_document(
    document_id: str,
    question: Dict[str, str],
    org_id: str = Depends(get_org_id)
):
    return await document_service.query_document(document_id, question, org_id)


# -------------------------
# DELETE DOCUMENT (ORG SAFE)
# -------------------------
@router.delete("/{document_id}")
async def delete_document(document_id: str, org_id: str = Depends(get_org_id)):
    return await document_service.delete_document(document_id, org_id)
