from fastapi import APIRouter, UploadFile, File
from typing import List, Dict
from backend.app.services.document_type_service import DocumentTypeService

router = APIRouter(prefix="/document-types", tags=["document-types"])
service = DocumentTypeService()


@router.get("/")
async def get_all_document_types():
    """Get all document types with stats."""
    return await service.get_all_document_types()


@router.get("/{doc_type_id}/documents")
async def get_documents_by_type(doc_type_id: str):
    """Get all documents for a given type."""
    return await service.get_documents_by_type(doc_type_id)


@router.post("/")
async def create_document_type(payload: Dict[str, str]):
    """Create a new document type."""
    return await service.create_document_type(payload["name"])


@router.put("/{doc_type_id}/fields")
async def update_fields(doc_type_id: str, payload: Dict[str, List[Dict]]):
    """Edit field definitions of a document type."""
    return await service.update_fields(doc_type_id, payload["fields"])


@router.post("/{doc_type_id}/upload")
async def upload_document_to_type(doc_type_id: str, file: UploadFile = File(...)):
    """Upload a file under a specific document type."""
    return await service.upload_document_to_type(doc_type_id, file)
