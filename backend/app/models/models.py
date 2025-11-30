
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from datetime import datetime

class TemplateIn(BaseModel):
    template_markdown: str

class DraftRequest(BaseModel):
    query: str

class FillTemplateRequest(BaseModel):
    template_id: str
    variables: Dict[str, str]

class PrefillRequest(BaseModel):
    template_id: str
    query: str

class GenerateQuestionsRequest(BaseModel):
    template_id: str
    filled_variables: Dict[str, str]

class DocumentOut(BaseModel):
    id: str
    status: str
    insights: Optional[Any] = None
    fullText: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime
    documentType: Optional[str] = None

    class Config:
        from_attributes = True