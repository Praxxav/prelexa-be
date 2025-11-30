"""
API routers package.
Contains all FastAPI route handlers organized by domain.
"""

from backend.app.api.documents import router as documents_router
from backend.app.api.templates import router as templates_router
# from backend.app.api.export import router as export_router

__all__ = ["documents_router", "templates_router", "export_router"]