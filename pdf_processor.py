# pdf_processor.py
import os
import requests
import fitz  # PyMuPDF
from langchain_core.documents import Document

def download_pdf(url: str, local_path: str) -> bool:
    """Downloads the PDF file if it doesn't exist locally."""
    if not os.path.exists(local_path):
        print(f"Downloading PDF from {url}...")
        try:
            # Add headers to mimic a web browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.aetnabetterhealth.com/illinois-medicaid/member-materials-forms.html' # Or the page where you found the link
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f"PDF downloaded to {local_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading PDF: {e}")
            return False
    else:
        print(f"PDF already exists at {local_path}. Skipping download.")
    return True

def extract_text_pypdf(pdf_path: str) -> list[Document]:
    """
    Extracts all text content from the PDF using PyMuPDF (Fitz).
    Returns a list of Langchain Document objects, each representing a page.
    """
    documents = []
    try:
        doc = fitz.open(pdf_path)
        print(f"Extracting text from {doc.page_count} pages...")
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text") # "text" extracts plain text
            
            # Create a Langchain Document object for each page
            # Metadata is important for source attribution in RAG
            documents.append(Document(
                page_content=text,
                metadata={"source": pdf_path, "page": page_num + 1}
            ))
            
        doc.close()
    
    except Exception as e:
        print(f"An error occurred during text extraction: {e}")
        return []
        
    return documents

if __name__ == "__main__":
    # Example usage if you run this script directly
    from config import PDF_URL, LOCAL_PDF_PATH
    if download_pdf(PDF_URL, LOCAL_PDF_PATH):
        docs = extract_text_pypdf(LOCAL_PDF_PATH)
        print(f"Extracted {len(docs)} documents.")
        if docs:
            print(f"First 200 chars of page 1: {docs[0].page_content[:200]}...")