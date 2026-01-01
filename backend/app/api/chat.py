from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from backend.app.services.chat_service import ChatService
from backend.app.utils.dependencies import get_org_id

router = APIRouter(prefix="/chat", tags=["chat"])

chat_service = ChatService()


# -------------------------
# REQUEST MODEL
# -------------------------
class ChatRequest(BaseModel):
    message: str
    document_id: Optional[str] = None


# -------------------------
# SEND MESSAGE
# -------------------------
@router.post("/message")
async def send_message(
    payload: ChatRequest,
    org_id: str = Depends(get_org_id)
):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message content is required.")

    return await chat_service.process_message(
        message=payload.message,
        document_id=payload.document_id,
        org_id=org_id
    )


# -------------------------
# CHAT HISTORY
# -------------------------
@router.get("/history")
async def get_chat_history(org_id: str = Depends(get_org_id)):
    return await chat_service.get_history(org_id)


# -------------------------
# CLEAR HISTORY
# -------------------------
@router.delete("/history")
async def clear_chat_history(org_id: str = Depends(get_org_id)):
    return await chat_service.clear_history(org_id)
