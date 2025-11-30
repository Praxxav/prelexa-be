import PyPDF2
import docx
import os
import logging
from typing import Optional

async def extract_text_from_file(file_path: str) -> Optional[str]:
    """
    Extract text content from various document formats.
    Currently supports PDF, DOCX, and TXT files.
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
        elif file_extension == '.docx':
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
            
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
                
        else:
            logging.error(f"Unsupported file type: {file_extension}")
            return None
            
    except Exception as e:
        logging.error(f"Error extracting text from file: {e}")
        return None