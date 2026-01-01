import logging
import json
import re
from typing import Dict, Optional
import google.generativeai as genai
from backend.db.database import db
from backend.core.config import settings

# --- Configure Gemini API ---
genai.configure(api_key=settings.GEMINI_API_KEY)

generation_config = {
    "temperature": 0.1,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 2048,
}

logger = logging.getLogger(__name__)


class DocumentTypeAnalyzer:
    """Agent for analyzing and detecting document types and fields."""

    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config=generation_config
        )

    async def _ensure_db_connection(self):
        """Ensure DB connection is active."""
        if not db.is_connected():
            await db.connect()

    # -------------------------------------------------------------------------
    # ✅ Robust JSON extractor that tolerates minor LLM formatting issues
    # -------------------------------------------------------------------------
    def _extract_json_from_text(self, text: str) -> dict:
        """Extract valid JSON from model output — tolerant to noise."""
        if not text or not text.strip():
            raise ValueError("Empty or invalid response text")

        text = text.strip()

        # Remove Markdown fences like ```json or ```
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

        # Remove trailing commas before } or ]
        text = re.sub(r",(\s*[}\]])", r"\1", text)

        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting inner JSON block
        json_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if json_match:
            inner = json_match.group(1)
            inner = re.sub(r",(\s*[}\]])", r"\1", inner)
            try:
                return json.loads(inner)
            except Exception as e:
                logger.warning(f"Inner JSON parse failed: {e}")

        # As last resort, sanitize nulls or missing values
        text = text.replace("None", "null").replace("NULL", "null")
        text = re.sub(r":\s*,", ": null,", text)
        try:
            return json.loads(text)
        except Exception as e:
            logger.warning(f"JSON parse failed: {e} | Raw text: {text[:300]}")
            raise ValueError("Failed to parse valid JSON from model output")

    # -------------------------------------------------------------------------
    async def _safe_generate(self, prompt: str) -> Optional[str]:
        """Safely call Gemini and return plain text."""
        try:
            response = await self.model.generate_content_async(prompt)
            if not response or not getattr(response, "candidates", None):
                return None

            parts = response.candidates[0].content.parts if response.candidates[0].content else []
            if not parts:
                return None

            text = getattr(parts[0], "text", None)
            return text.strip() if text else None

        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            return None

    # -------------------------------------------------------------------------
    async def analyze_document(self, content: str) -> Dict:
        """Detect document type and extract key fields."""
        try:
            # --- Step 1: Classify document type ---
            type_prompt = f"""
            You are an expert document classification system.

                Analyze the document content and determine the MOST LIKELY document type.

                Rules:
                - If the document matches a known type, be specific (Resume, Invoice, NDA, Offer Letter, Bank Statement, ID Proof, Certificate, Notice, Form, Report, Letter, etc.)
                - If it does NOT clearly match any known type, return "Generic Document"
                - NEVER return an empty or missing field
                - Confidence must be between 0.0 and 1.0

                Document Content (first 2000 chars):
                {content[:2000]}

                Return ONLY valid JSON in this exact structure:
                {{
                "document_type": "Resume | Invoice | Agreement | Certificate | Letter | Generic Document",
                "confidence": 0.0,
                "category": "Legal | Finance | HR | Education | Personal | Business | Other",
                "key_identifiers": ["names", "dates", "amounts", "ids", "emails"]
                }}
            """
            type_text = await self._safe_generate(type_prompt)
            type_data = self._extract_json_from_text(type_text) if type_text else {}

            # --- Step 2: Extract fields ---
            fields_prompt = f"""
           You are an expert data extraction system.

                The document type is: "{type_data.get('document_type', 'Generic Document')}"

                Your job:
                - ALWAYS extract meaningful fields
                - The fields array MUST contain at least 5 items
                - If a value is not clearly found, set value to null
                - Use snake_case for field names
                - Fields MUST be relevant to the document OR generic metadata

                If the document is unclear or unstructured, extract GENERIC fields such as:
                - document_summary
                - primary_entity
                - secondary_entity
                - important_dates
                - reference_numbers
                - contact_information
                - detected_keywords
                - raw_excerpt

                Return ONLY valid JSON in this structure:
                {{
                "fields": [
                    {{
                    "name": "snake_case_name",
                    "value": "string | number | date | null",
                    "required": false,
                    "description": "What this field represents"
                    }}
                ]
                }}

                Failure to return at least 5 fields is NOT acceptable.
            """
            fields_text = await self._safe_generate(fields_prompt)
            fields_data = self._extract_json_from_text(fields_text) if fields_text else {}

            # --- Merge and return ---
            return {
                "document_type": type_data.get("document_type", "Unknown"),
                "confidence": type_data.get("confidence", 0.0),
                "category": type_data.get("category", "Unknown"),
                "key_identifiers": type_data.get("key_identifiers", []),
                "fields": fields_data.get("fields", []),
            }

        except Exception as e:
            logger.error(f"Error analyzing document: {e}", exc_info=True)
            return self._get_default_analysis()

    # -------------------------------------------------------------------------
    def _get_default_analysis(self) -> Dict:
        """Default response when analysis fails."""
        return {
            "document_type": "Generic Document",
            "confidence": 0.0,
            "category": "Unknown",
            "key_identifiers": [],
            "fields": [],
        }

    # -------------------------------------------------------------------------
    async def get_or_create_document_type(self, analysis_result: Dict) -> Optional[str]:
        """Create or fetch DocumentType entry in DB."""
        await self._ensure_db_connection()

        try:
            doc_type_name = analysis_result.get("document_type", "Unknown")
            if doc_type_name == "Unknown" or analysis_result.get("confidence", 0) < 0.3:
                return None

            existing = await db.documenttype.find_first(where={"name": doc_type_name})
            if existing:
                return existing.id

            new_type = await db.documenttype.create(
                data={
                    "name": doc_type_name,
                    "category": analysis_result.get("category", "Unknown"),
                    "description": f"Auto-detected document type (confidence {analysis_result.get('confidence', 0):.2f})",
                    "fields": json.dumps(analysis_result.get("fields", [])),
                    "metadata": json.dumps({
                        "confidence": analysis_result.get("confidence", 0),
                        "key_identifiers": analysis_result.get("key_identifiers", []),
                        "auto_detected": True,
                    }),
                }
            )
            return new_type.id

        except Exception as e:
            logger.error(f"Error saving document type: {e}", exc_info=True)
            return None
