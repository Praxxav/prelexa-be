from typing import Dict, List
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, org_id: str, websocket: WebSocket):
        await websocket.accept()
        if org_id not in self.active_connections:
            self.active_connections[org_id] = []
        self.active_connections[org_id].append(websocket)

    def disconnect(self, org_id: str, websocket: WebSocket):
        self.active_connections[org_id].remove(websocket)
        if not self.active_connections[org_id]:
            del self.active_connections[org_id]

    async def broadcast(self, org_id: str, message: dict):
        for ws in self.active_connections.get(org_id, []):
            await ws.send_json(message)
connection_manager = ConnectionManager()
