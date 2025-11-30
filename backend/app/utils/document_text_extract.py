import logging
import pytesseract
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
from PIL import Image
import os

# Optional: setup basic logging
logging.basicConfig(level=logging.INFO)

def extract_text_with_pdfminer(file_path: str) -> str:
    """
    Extract text from a PDF using pdfminer.six
    """
    try:
        return extract_text(file_path) or ""
    except Exception as e:
        logging.error(f"PDFMiner failed to extract text: {e}")
        return ""

import docx

async def extract_text_from_file(file_path: str) -> str:
    try:
        extension = os.path.splitext(file_path)[-1].lower()

        if extension == ".pdf":
            text = extract_text_with_pdfminer(file_path)
            if not text.strip():
                logging.info("Falling back to OCR for PDF...")
                pages = convert_from_path(file_path)
                return "".join(pytesseract.image_to_string(page) for page in pages)
            return text

        elif extension in [".png", ".jpg", ".jpeg"]:
            return pytesseract.image_to_string(Image.open(file_path))

        elif extension == ".docx":
            doc = docx.Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])

        elif extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        else:
            logging.warning(f"Unsupported file type: {extension}")
            return ""

    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return ""
