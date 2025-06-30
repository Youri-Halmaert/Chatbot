import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path):
    """Extrait le texte d'un fichier PDF."""
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def extract_texts_from_folder(folder_path):
    """Extrait le texte de tous les PDFs d'un dossier."""
    pdf_texts = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            path = os.path.join(folder_path, filename)
            pdf_texts[filename] = extract_text_from_pdf(path)
    return pdf_texts
