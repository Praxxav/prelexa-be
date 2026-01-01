from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
import os

from backend.db.database import db
from backend.db.database import lifespan
from backend.app.api.documents import router as documents_router
from backend.app.api.templates import router as templates_router    
from backend.app.api.export import router as export_router
from backend.app.api.document_variables import router as document_variables_router
from backend.app.api.chat import router as chat_router
# Create FastAPI app
app = FastAPI(
    title="Prelexa AI Document Analysis API",
    description="API for AI-powered legal document review and insight generation.",
    version="0.2.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Include routers
app.include_router(documents_router, tags=["documents"])
app.include_router(templates_router, tags=["templates"])
app.include_router(export_router, prefix="/export", tags=["export"])
app.include_router(document_variables_router, tags=["Document Variables"])
app.include_router(chat_router)
# app.include_router(export_router, prefix="/api")


@app.on_event("startup")
async def startup():
    """Connect to database on startup."""
    try:
        logging.info("Attempting to connect to the database...")
        await db.connect()
        logging.info("Database connection successful.")
    except Exception as e:
        logging.error("--- DATABASE CONNECTION FAILED ---")
        logging.error("Could not connect to the database. Please check:")
        logging.error("1. Your `DATABASE_URL` in the .env file is correct.")
        logging.error("2. The database server is running and accessible.")
        logging.error("3. Network/firewall is not blocking port 5432.")
        logging.error(f"Error: {e}")
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown():
    """Disconnect from database on shutdown."""
    await db.disconnect()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Intelligent Document Analysis API",
        "version": "0.2.0",
        "endpoints": {
            "documents": "/documents",
            "templates": "/templates",
            "export": "/export"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "database": "connected"}