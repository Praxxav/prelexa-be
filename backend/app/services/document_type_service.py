from fastapi import UploadFile, HTTPException
from typing import Dict, List
import os
import json
import logging
from prisma import Prisma

UPLOAD_DIR = "uploaded_document_types"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class DocumentTypeService:
    """Service for managing document types and related uploads."""

    def __init__(self):
        self.db = Prisma()

    async def get_all_document_types(self):
        """Fetch all document types with basic stats."""
        try:
            doc_types = await self.db.documenttype.find_many(
                include={"documents": True}
            )

            formatted = []
            for t in doc_types:
                uploaded = len(t.documents)
                review_pending = len([d for d in t.documents if d.status == "review_pending"])
                approved = len([d for d in t.documents if d.status == "approved"])

                formatted.append({
                    "id": t.id,
                    "name": t.name,
                    "uploaded": uploaded,
                    "review_pending": review_pending,
                    "approved": approved
                })

            return {"document_types": formatted}

        except Exception as e:
            logging.error(f"Error fetching document types: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_documents_by_type(self, doc_type_id: str):
        """Return all documents belonging to a specific document type."""
        doc_type = await self.db.documenttype.find_unique(
            where={"id": doc_type_id},
            include={"documents": True}
        )
        if not doc_type:
            raise HTTPException(status_code=404, detail="Document type not found")

        return {
            "document_type": doc_type.name,
            "documents": doc_type.documents
        }

    async def create_document_type(self, name: str):
        """Create a new document type."""
        existing = await self.db.documenttype.find_first(where={"name": name})
        if existing:
            raise HTTPException(status_code=400, detail="Document type already exists")

        new_type = await self.db.documenttype.create(
            data={"name": name, "fields": json.dumps([])}
        )

        return {"message": "Document type created successfully", "document_type": new_type}

    async def update_fields(self, doc_type_id: str, fields: List[Dict]):
        """Update the field definitions of a document type."""
        doc_type = await self.db.documenttype.find_unique(where={"id": doc_type_id})
        if not doc_type:
            raise HTTPException(status_code=404, detail="Document type not found")

        await self.db.documenttype.update(
            where={"id": doc_type_id},
            data={"fields": json.dumps(fields)}
        )

        return {"message": "Fields updated successfully", "document_type_id": doc_type_id}

    async def upload_document_to_type(self, doc_type_id: str, file: UploadFile):
        """Upload a document under a specific type."""
        doc_type = await self.db.documenttype.find_unique(where={"id": doc_type_id})
        if not doc_type:
            raise HTTPException(status_code=404, detail="Document type not found")

        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in [".pdf", ".docx", ".txt"]:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        file_path = os.path.join(UPLOAD_DIR, f"{doc_type_id}_{file.filename}")
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        new_doc = await self.db.document.create(
            data={
                "documentTypeId": doc_type_id,
                "filename": file.filename,
                "status": "uploaded",
                "filePath": file_path,
            }
        )

        return {"message": "Document uploaded successfully", "document": new_doc}

