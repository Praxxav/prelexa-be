# backend/app/agent/document_agent.py

from backend.app.agent.base_agent import SimpleAgent
from backend.core.config import settings
DOCUMENT_ANALYZER_PROMPT = """
You are a universal document intelligence engine.

You must NEVER fail, NEVER return empty fields, and NEVER skip extraction.

Your task:
1. Identify the document type.
   - If confident, use a specific type (Invoice, Resume, Certificate, Form, ID Proof, Agreement, Letter, Report, etc.)
   - If uncertain, use: "Generic Document"

2. Generate a short, human-readable title.

3. Extract structured fields using this schema:

{
  "title": "string",
  "document_type": "string",
  "fields": [
    {
      "name": "string_identifier",
      "label": "Human readable label",
      "value": "string | number | date | null",
      "type": "string | number | date | boolean",
      "confidence": 0.0 - 1.0,
      "editable": true | false
    }
  ]
}

IMPORTANT RULES:
- Fields array MUST contain at least 5 items.
- If the document is unstructured, extract:
  - Names
  - Dates
  - Emails
  - Phone numbers
  - IDs
  - Headings
  - Table rows as key-value pairs
- If values are missing, set value = null.
- If the document is a form or table, infer column headers as field names.
- If nothing is identifiable, add:
  - summary
  - raw_text_excerpt
  - detected_keywords

Return ONLY valid JSON.
No markdown. No explanation.
"""


document_agent = SimpleAgent(
    name="DocumentAgent",
    role="Extracts structured information and variables from uploaded documents.",
    api_key=settings.GEMINI_API_KEY,
    system_prompt=DOCUMENT_ANALYZER_PROMPT,
    model="gemini-2.5-flash",
)


async def analyze_document_text(text: str) -> dict:
    """
    Given full document text, returns structured analysis:
    { title, document_type, fields[] }
    """
    prompt = f"""
Analyze the following document text and extract all key structured information.

Document Text:
{text}
"""
    result = await document_agent.process(prompt)

    import json, re

    try:
        match = re.search(r"\{.*\}", result, re.DOTALL)
        json_str = match.group(0) if match else result
        parsed = json.loads(json_str)

        return {
            "title": parsed.get("title", "Untitled Document"),
            "document_type": parsed.get("document_type", "unknown"),
            "fields": parsed.get("fields", []),
        }
    except Exception as e:
        return {"error": f"Failed to parse: {e}", "raw_output": result}
