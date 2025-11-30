import os
import json
import logging
from fastapi import UploadFile, BackgroundTasks, HTTPException

from backend.db.database import db
from backend.app.agent.document_agent import analyze_document_text
from backend.app.utils.uploads import UPLOAD_DIR
from backend.app.utils.document_text_extract import extract_text_from_file
from backend.app.services.document_variable_service import DocumentVariableService

logger = logging.getLogger(__name__)


# ---------------------------
# JSON SERIALIZER (Fix UI)
# ---------------------------
def serialize_document(doc):
    """Convert JSON fields stored as TEXT into real JSON objects"""
    data = doc.dict()

    # Convert JSON strings → JSON objects
    if data.get("metadata"):
        try:
            data["metadata"] = json.loads(data["metadata"])
        except:
            pass

    if data.get("insights"):
        try:
            data["insights"] = json.loads(data["insights"])
        except:
            pass

    if data.get("fields"):
        try:
            data["fields"] = json.loads(data["fields"])
        except:
            pass

    return data



class DocumentService:

    # ---------------------------
    # UPLOAD DOCUMENT — ORG SAFE
    # ---------------------------
    async def upload_document(self, file: UploadFile, background_tasks: BackgroundTasks, org_id: str):
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"]:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        doc = await db.document.create(
            data={
                "status": "uploaded",
                "filePath": file_path,
                "orgId": org_id,
            }
        )

        background_tasks.add_task(
            self._process_document_background, doc.id, file_path, ext
        )

        return {"message": "Document uploaded successfully", "document_id": doc.id}

    # ---------------------------
    # BACKGROUND AI PROCESS
    # ---------------------------
    async def _process_document_background(self, doc_id: str, file_path: str, ext: str):
        try:
            await db.document.update(
                where={"id": doc_id},
                data={"status": "processing"}
            )

            text = await extract_text_from_file(file_path)
            if not text.strip():
                await db.document.update(
                    where={"id": doc_id},
                    data={"status": "failed"}
                )
                return

            analysis = await analyze_document_text(text)

            if "error" in analysis:
                await db.document.update(
                    where={"id": doc_id},
                    data={"status": "failed"}
                )
                return

            await db.document.update(
                where={"id": doc_id},
                data={
                    "status": "completed",
                    "fullText": text,
                    "documentType": analysis.get("document_type", "Unknown"),
                    "metadata": json.dumps({"title": analysis.get("title", "Untitled")}),
                    "insights": json.dumps(analysis),
                },
            )

            fields = analysis.get("fields", [])
            if fields:
                await DocumentVariableService.bulk_create_variables(
                    doc_id,
                    [
                        {
                            "name": f["name"],
                            "value": str(f.get("value", "")),
                            "confidence": f.get("confidence", 1.0),
                            "editable": f.get("editable", True),
                        }
                        for f in fields
                    ],
                )

        except Exception as e:
            logger.error(f"[{doc_id}] Processing failed: {e}", exc_info=True)
            await db.document.update(
                where={"id": doc_id},
                data={"status": "failed"}
            )

    # ---------------------------
    # GET ALL DOCUMENTS — ORG SAFE
    # ---------------------------
    async def get_all_documents(self, org_id: str):
        docs = await db.document.find_many(
            where={"orgId": org_id},
            order={"createdAt": "desc"},
        )

        # FIX: convert JSON strings → real JSON for UI
        return {
            "success": True,
            "data": [serialize_document(doc) for doc in docs]
        }

    # ---------------------------
    # GET FIELDS — ORG SAFE
    # ---------------------------
    async def get_document_fields(self, doc_id: str, org_id: str):
        doc = await db.document.find_unique(where={"id": doc_id})
        if not doc or doc.orgId != org_id:
            raise HTTPException(status_code=403, detail="Unauthorized")

        variables = await db.documentvariable.find_many(
            where={"documentId": doc_id}
        )
        return {"success": True, "data": variables}

    # ---------------------------
    # UPDATE FIELDS — ORG SAFE
    # ---------------------------
    async def update_document_fields(self, doc_id: str, fields: dict, org_id: str):
        doc = await db.document.find_unique(where={"id": doc_id})
        if not doc or doc.orgId != org_id:
            raise HTTPException(status_code=403, detail="Unauthorized")

        for key, value in fields.items():
            await db.documentvariable.update_many(
                where={"documentId": doc_id, "name": key},
                data={"value": value},
            )
        return {"success": True, "message": "Fields updated successfully"}

    # ---------------------------
    # GET FILE — ORG SAFE
    # ---------------------------
    async def get_document_file_path(self, doc_id: str, org_id: str):
        doc = await db.document.find_unique(where={"id": doc_id})
        if not doc or doc.orgId != org_id:
            raise HTTPException(status_code=403, detail="Unauthorized")

        return doc.filePath

    # ---------------------------
    # STATUS — ORG SAFE
    # ---------------------------
    async def get_processing_status(self, doc_id: str, org_id: str):
        doc = await db.document.find_unique(where={"id": doc_id})
        if not doc or doc.orgId != org_id:
            raise HTTPException(status_code=403, detail="Unauthorized")

        return {"document_id": doc_id, "status": doc.status}

    # ---------------------------
    # INSIGHTS — ORG SAFE
    # ---------------------------
    async def get_document_insights(self, doc_id: str, org_id: str):
        doc = await db.document.find_unique(where={"id": doc_id})
        if not doc or doc.orgId != org_id:
            raise HTTPException(status_code=403, detail="Unauthorized")

        return json.loads(doc.insights or "{}")

    # ---------------------------
    # QUERY — ORG SAFE
    # ---------------------------
    async def query_document(self, doc_id: str, question: dict, org_id: str):
        doc = await db.document.find_unique(where={"id": doc_id})
        if not doc or doc.orgId != org_id:
            raise HTTPException(status_code=403, detail="Unauthorized")

        return await analyze_document_text(doc.fullText, question.get("question"))

    # ---------------------------
    # DELETE — ORG SAFE
    # ---------------------------
    async def delete_document(self, doc_id: str, org_id: str):
        doc = await db.document.find_unique(where={"id": doc_id})
        if not doc or doc.orgId != org_id:
            raise HTTPException(status_code=403, detail="Unauthorized")

        if doc.filePath and os.path.exists(doc.filePath):
            os.remove(doc.filePath)

        await db.documentvariable.delete_many(where={"documentId": doc_id})
        await db.document.delete(where={"id": doc_id})

        return {"success": True, "message": "Document deleted successfully"}
