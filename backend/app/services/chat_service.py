import logging
from typing import Optional
from datetime import datetime

from backend.app.agent.chat_agent import DocumentChatAgent
from backend.core.config import settings
from backend.db.database import db

logger = logging.getLogger(__name__)

# ✅ Correct agent instantiation (ONCE)
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
        await self._save_message(
            org_id=org_id,
            document_id=document_id,
            content=message,
            role="user"
        )

        # -------------------------
        # DOCUMENT CHAT
        # -------------------------
        if document_id:
            response = await document_chat_agent.process(
                document_id=document_id,
                question=message,
                org_id=org_id,
            )

            await self._save_message(
                org_id=org_id,
                document_id=document_id,
                content=response,
                role="assistant"
            )

            return {
                "type": "document_chat",
                "document_id": document_id,
                "answer": response
            }

        # -------------------------
        # NO DOCUMENT (optional)
        # -------------------------
        return {
            "type": "general_chat",
            "answer": "No document selected for chat."
        }

    # -------------------------
    # CHAT HISTORY
    # -------------------------
    async def get_history(self, org_id: str):
        chats = await db.chatmessage.find_many(
            where={"orgId": org_id},
            order={"createdAt": "asc"}
        )
        return {"success": True, "data": chats}

    async def clear_history(self, org_id: str):
        await db.chatmessage.delete_many(where={"orgId": org_id})
        return {"success": True, "message": "Chat history cleared."}

    # -------------------------
    # SAVE MESSAGE
    # -------------------------
    async def _save_message(
        self,
        org_id: str,
        document_id: Optional[str],
        content: str,
        role: str
    ):
        try:
            return await db.chatmessage.create(
                data={
                    "orgId": org_id,
                    "documentId": document_id,
                    "content": str(content),
                    "role": role,
                    "createdAt": datetime.utcnow(),
                }
            )
        except Exception as e:
            logger.warning(f"Failed to save chat message: {e}")
