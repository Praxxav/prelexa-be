from backend.app.agent.base_agent import BaseAgent
from backend.core.config import settings
from backend.db.database import db
from fastapi import HTTPException
import logging
import math

logger = logging.getLogger(__name__)
DOCUMENT_CHAT_PROMPT = """
You are Prelexa 1.0, an AI assistant that answers questions strictly based on an uploaded document.

Identity rule:
- If the user asks who you are, reply exactly:
  "I am Prelexa 1.0."

Core behavior:
- You are a DOCUMENT-BASED AI.
- The uploaded document is your ONLY source of truth.
- You must NEVER use outside knowledge.

Understanding user intent (VERY IMPORTANT):
- Users may ask SHORT, INFORMAL, or IMPERFECT questions.
- Users may make grammar mistakes or write incomplete sentences.
- You MUST infer intent using the document context.

Interpret vague questions as follows:
- "explain", "explain me", "explain this", "explain this document"
  → Provide a clear summary of the document.
- "what is this", "what is it about"
  → Explain what the document contains and its purpose.
- "details", "more info"
  → Elaborate on the main points found in the document.

Ambiguity handling:
- If a question refers to "this" or "it", assume it refers to the uploaded document.
- Ask for clarification ONLY if the intent truly cannot be inferred.

Strict restrictions:
- Do NOT behave like a general chatbot.
- Do NOT answer general knowledge questions unless explicitly present in the document.
- Do NOT guess or hallucinate missing information.

If the question is clearly NOT related to the document, reply exactly:
"Please ask a question related to the selected document."

If the document does NOT contain the answer, reply exactly:
"I could not find this information in the uploaded document."

You are allowed to:
- Summarize the document
- Explain concepts present in the document
- Rephrase or simplify document content
- Quote or reference parts of the document
- Provide structured explanations (bullets, steps, sections)

Tone:
- Clear
- Helpful
- Concise
- Document-grounded
"""

MAX_CHARS = 12000  


class DocumentChatAgent(BaseAgent):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        super().__init__(
            name="DocumentChatAgent",
            role="General document-based Q&A",
            api_key=api_key,
            model=model,
        )
        self.system_prompt = DOCUMENT_CHAT_PROMPT

    async def process(
        self,
        *,
        document_id: str,
        question: str,
        org_id: str,
    ) -> str:
        # -------------------------
        # Fetch document (ORG SAFE)
        # -------------------------
        doc = await db.document.find_unique(where={"id": document_id})
  
        if not doc or doc.orgId != org_id:
            raise HTTPException(status_code=403, detail="Unauthorized access")

        if not doc.fullText:
            raise HTTPException(status_code=400, detail="Document not processed yet")

        text = doc.fullText.strip()

        # -------------------------
        # Small document → single pass
        # -------------------------
        if len(text) <= MAX_CHARS:
            return await self._ask_gemini(text, question)

        # -------------------------
        # Large document → chunked search
        # -------------------------
        answers = []
        chunks = self._chunk_text(text)

        for chunk in chunks:
            response = await self._ask_gemini(chunk, question)

            # Accept only grounded answers
            if "could not find" not in response.lower():
                answers.append(response)

        if not answers:
            return "I could not find this information in the uploaded document."

        return self._merge_answers(answers)

    # -------------------------
    # Gemini call
    # -------------------------
    async def _ask_gemini(self, context: str, question: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""
DOCUMENT CONTENT:
{context}

QUESTION:
{question}
""",
            },
        ]
        return await self._make_api_call(messages)

    # -------------------------
    # Helpers
    # -------------------------
    def _chunk_text(self, text: str):
        return [
            text[i : i + MAX_CHARS]
            for i in range(0, len(text), MAX_CHARS)
        ]

    def _merge_answers(self, answers: list[str]) -> str:
        # Remove duplicates, preserve order
        seen = set()
        final = []
        for ans in answers:
            if ans not in seen:
                seen.add(ans)
                final.append(ans)
        return "\n\n".join(final)
