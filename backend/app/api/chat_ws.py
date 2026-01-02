# backend/app/api/chat_ws.py
from fastapi import APIRouter, WebSocket, Depends
from backend.app.websocket.websocket_manager import connection_manager as manager
from backend.app.utils.dependencies import get_org_id

router = APIRouter()

@router.websocket("/chat/ws")
async def chat_ws(websocket: WebSocket, org_id: str = Depends(get_org_id)):
    await manager.connect(org_id, websocket)
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except:
        manager.disconnect(org_id, websocket)
