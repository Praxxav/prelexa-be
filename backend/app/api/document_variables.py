from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.app.services.document_variable_service import DocumentVariableService

router = APIRouter(prefix="/document-variables", tags=["Document Variables"])

class VariableCreate(BaseModel):
    document_id: str
    name: str
    value: Optional[str] = None
    confidence: Optional[float] = None

class VariableUpdate(BaseModel):
    value: Optional[str] = None

@router.post("/")
async def create_variable(payload: VariableCreate):
    try:
        variable = await DocumentVariableService.create_variable(
            document_id=payload.document_id,
            name=payload.name,
            value=payload.value,
            confidence=payload.confidence,
        )
        return {"success": True, "data": variable}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}")
async def get_variables(document_id: str):
    try:
        variables = await DocumentVariableService.get_variables(document_id)
        return {"success": True, "data": variables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{variable_id}")
async def update_variable(variable_id: str, payload: VariableUpdate):
    try:
        updated = await DocumentVariableService.update_variable(
            variable_id=variable_id,
            value=payload.value,
        )
        return {"success": True, "data": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{variable_id}")
async def delete_variable(variable_id: str):
    try:
        deleted = await DocumentVariableService.delete_variable(variable_id)
        return {"success": True, "data": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
