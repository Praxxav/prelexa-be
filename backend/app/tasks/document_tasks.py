import asyncio
import json
import logging
import os
import re
import PyPDF2
import docx
from backend.db.database import db
from backend.app.agent import law as law_agents
from backend.app.agent.router import classifier_agent
from backend.app.utils.document_text_extract import extract_text_from_file  # OCR + pdfminer
from backend.app.services.document_variable_service import DocumentVariableService
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

def extract_insight_tags(text_content: str, entities: dict):
    """
    Generate quick insight tags based on keywords or entity keys.
    Helps frontend display useful labels (e.g., 'Invoice', 'Legal', 'Offer Letter').
    """
    try:
        tags = set()

        # Check document keywords
        lowered = text_content.lower()
        if "invoice" in lowered or "amount" in lowered:
            tags.add("finance")
        if "agreement" in lowered or "terms" in lowered:
            tags.add("legal")
        if "resume" in lowered or "curriculum vitae" in lowered:
            tags.add("hr")
        if "report" in lowered:
            tags.add("report")

        # Add tags based on entities
        if isinstance(entities, dict):
            tags.update([k.lower() for k in entities.keys() if isinstance(k, str) and len(k) < 25])

        return list(tags)
    except Exception as e:
        logger.warning(f"Failed to extract insight tags: {e}")
        return []


async def process_document_in_background(document_id: str, file_path: str, file_extension: str):
    """
    Background task that processes the uploaded document.
    1. Extracts text (with OCR fallback)
    2. Classifies document type
    3. Generates summary and entities using AI agents
    4. Saves extracted variables (entities)
    5. Updates DB with final structured insights
    """
    try:
        await db.document.update(where={"id": document_id}, data={"status": "processing"})
        logger.info(f"[{document_id}] Processing started...")

        # --- STEP 1: Extract text ---
        logger.info(f"[{document_id}] Extracting text from file...")
        text_content = await safe_extract_text(file_path, file_extension)

        if not text_content or len(text_content.strip()) < 20:
            logger.error(f"[{document_id}] Text extraction failed or empty.")
            await db.document.update(where={"id": document_id}, data={"status": "failed"})
            return

        # --- STEP 2: Classify document type ---
        logger.info(f"[{document_id}] Classifying document type...")
        try:
            doc_type_raw = await classifier_agent.process(text_content)
            doc_type = doc_type_raw.strip().lower()
        except Exception as e:
            logger.error(f"[{document_id}] Classification failed: {e}")
            doc_type = "unknown"

        # Choose appropriate agent set based on type (future scalability)
        summarizer = law_agents.summarizer_agent
        entity_extractor = law_agents.entity_extractor_agent

        # --- STEP 3: Generate insights concurrently ---
        logger.info(f"[{document_id}] Running summarizer + entity extraction...")
        try:
            summary_task = summarizer.process(text_content)
            entities_task = entity_extractor.process(text_content)
            summary, entities_raw = await asyncio.gather(summary_task, entities_task)
        except Exception as e:
            logger.error(f"[{document_id}] Insight generation failed: {e}")
            summary = "Failed to generate summary"
            entities_raw = "{}"

        # --- STEP 4: Parse entities JSON safely ---
        entities = safe_parse_json(entities_raw)

        # --- STEP 5: Save extracted entities as variables ---
        try:
            if isinstance(entities, dict) and entities:
                variables = []
                for key, value in entities.items():
                    if isinstance(value, dict):
                        variables.append({
                            "name": key,
                            "value": json.dumps(value, ensure_ascii=False),
                            "confidence": value.get("confidence") if isinstance(value, dict) else None
                        })
                    else:
                        variables.append({
                            "name": key,
                            "value": str(value),
                            "confidence": None
                        })

                if variables:
                    await DocumentVariableService.bulk_create_variables(document_id, variables)
                    logger.info(f"[{document_id}] ✅ Saved {len(variables)} variables successfully.")
                else:
                    logger.info(f"[{document_id}] No variables found to save.")
            else:
                logger.warning(f"[{document_id}] Entity data invalid or empty.")
        except Exception as e:
            logger.error(f"[{document_id}] Failed to save variables: {e}")

        # --- ✅ STEP 6: Update DB with structured insights ---
        structured_insights = {
            "document_metadata": {
                "id": document_id,
                "file_name": os.path.basename(file_path),
                "file_type": file_extension,
                "document_type": doc_type,
                "page_count": None,
            },
            "summary": {
                "text": summary if summary else "No summary available.",
                "length": len(summary) if isinstance(summary, str) else 0,
            },
            "entities": entities if isinstance(entities, dict) else {"raw_output": str(entities)},
            "analysis": {
                "confidence_level": "medium" if entities else "low",
                "insight_tags": extract_insight_tags(text_content, entities),
            },
        }

        await db.document.update(
            where={"id": document_id},
            data={
                "status": "completed",
                "insights": json.dumps(structured_insights, ensure_ascii=False, indent=2),
                "documentType": doc_type,
                "fullText": text_content,
            },
        )

        logger.info(f"[{document_id}] ✅ Document processing completed successfully.")

    except Exception as e:
        logger.error(f"[{document_id}] ❌ Fatal error during processing: {e}", exc_info=True)
        await db.document.update(where={"id": document_id}, data={"status": "failed"})
        

# --- Helper: Robust text extraction ---
async def safe_extract_text(file_path: str, file_extension: str) -> str:
    """
    Extract text from PDFs, DOCX, TXT, or images.
    Includes OCR fallback for image-based PDFs.
    """
    try:
        if file_extension == ".pdf":
            # Try normal text extraction first
            text = await extract_text_from_file(file_path)
            if not text.strip():
                logger.info("No text found via pdfminer — falling back to OCR...")
                pages = convert_from_path(file_path)
                ocr_text = ""
                for page in pages:
                    ocr_text += pytesseract.image_to_string(page)
                return ocr_text
            return text

        elif file_extension == ".docx":
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])

        elif file_extension in [".png", ".jpg", ".jpeg"]:
            image = Image.open(file_path)
            return pytesseract.image_to_string(image)

        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        else:
            logger.warning(f"Unsupported file type for extraction: {file_extension}")
            return ""

    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""


# --- Helper: Safe JSON parsing for model output ---
def safe_parse_json(raw_output):
    """
    Extract and safely parse JSON from AI output.
    Supports raw dicts, JSON strings, and fenced JSON blocks.
    """
    try:
        if isinstance(raw_output, dict):
            return raw_output

        if not isinstance(raw_output, str):
            raw_output = str(raw_output)

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
        json_str = json_match.group(1) if json_match else raw_output.strip()
        return json.loads(json_str)
    except Exception as e:
        logger.warning(f"Failed to parse JSON entities: {e}")
        return {"error": "Invalid JSON", "raw_output": raw_output}
