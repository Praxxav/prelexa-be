# backend/app/agent/document_agent.py

from backend.app.agent.base_agent import SimpleAgent
from backend.core.config import settings

DOCUMENT_ANALYZER_PROMPT = """
You are a document analysis expert.

Given the full text of a document, you must:
1. Identify what type of document it is (invoice, agreement, offer letter, ID proof, purchase order, etc.).
2. Generate a clear title summarizing the document (e.g., "Vendor Agreement with XYZ Pvt Ltd").
3. Extract all important structured fields (variables) with this schema:

{
  "fields": [
    {
      "name": "party_name",
      "label": "Party Name",
      "value": "XYZ Pvt Ltd",
      "type": "string",
      "confidence": 0.95,
      "editable": true
    },
    {
      "name": "agreement_date",
      "label": "Agreement Date",
      "value": "12 March 2024",
      "type": "date",
      "confidence": 0.9,
      "editable": true
    }
  ]
}

Return ONLY valid JSON (no Markdown, no explanation). Include fields even if values are missing, if they’re relevant for this document type.
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
