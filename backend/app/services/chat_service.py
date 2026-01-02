import logging
from typing import Optional
from datetime import datetime

from backend.app.agent.chat_agent import DocumentChatAgent
from backend.core.config import settings
from backend.db.database import db
from backend.app.websocket.websocket_manager import ConnectionManager as manager

logger = logging.getLogger(__name__)

document_chat_agent = DocumentChatAgent(
    api_key=settings.GEMINI_API_KEY,
    model="gemini-2.5-flash"
)


class ChatService:

    async def process_message(
        self,
        message: str,
        org_id: str,
        document_id: Optional[str] = None
    ):
        # Save user message
        user_msg = await self._save_message(
            org_id=org_id,
            document_id=document_id,
            content=message,
            role="user"
        )

        # Broadcast user message
        await manager.broadcast(org_id, {
            "type": "message",
            "id": user_msg.id,
            "role": "user",
            "content": message,
            "documentId": document_id,
            "createdAt": user_msg.createdAt.isoformat()
        })

        if document_id:
            response = await document_chat_agent.process(
                document_id=document_id,
                question=message,
                org_id=org_id,
            )

            assistant_msg = await self._save_message(
                org_id=org_id,
                document_id=document_id,
                content=response,
                role="assistant"
            )

            await manager.broadcast(org_id, {
                "type": "message",
                "id": assistant_msg.id,
                "role": "assistant",
                "content": response,
                "documentId": document_id,
                "createdAt": assistant_msg.createdAt.isoformat()
            })

            return {
                "type": "document_chat",
                "document_id": document_id,
                "answer": response
            }

        return {
            "type": "general_chat",
            "answer": "No document selected for chat."
        }

    # -------------------------
    # CHAT HISTORY (FIXED)
    # -------------------------
    async def get_history(self, org_id: str):
        chats = await db.chatmessage.find_many(
            where={"orgId": org_id},
            order={"createdAt": "asc"}
        )

        # Collect document IDs
        document_ids = {c.documentId for c in chats if c.documentId}

        documents = {}

        if document_ids:
            docs = await db.document.find_many(
                where={"id": {"in": list(document_ids)}}
            )

            # Build id -> title map
            for d in docs:
                title = "Untitled Document"
                if d.metadata and isinstance(d.metadata, dict):
                    title = d.metadata.get("title", title)
                documents[d.id] = title

        data = []
        for c in chats:
            data.append({
                "id": c.id,
                "role": c.role,
                "content": c.content,
                "documentId": c.documentId,
                "createdAt": c.createdAt.isoformat(),
                "documentTitle": documents.get(
                    c.documentId, "Untitled Document"
                ),
            })

        return {"success": True, "data": data}

    async def clear_history(self, org_id: str):
        await db.chatmessage.delete_many(where={"orgId": org_id})
        return {"success": True}

    async def _save_message(
        self,
        org_id: str,
        document_id: Optional[str],
        content: str,
        role: str
    ):
        return await db.chatmessage.create(
            data={
                "orgId": org_id,
                "documentId": document_id,
                "content": content,
                "role": role,
                "createdAt": datetime.utcnow(),
            }
        )
