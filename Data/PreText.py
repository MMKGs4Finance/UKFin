import os
import re
import requests
from bs4 import BeautifulSoup
import PyPDF2

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_text_from_url(url: str, timeout: int = 20) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        main = soup.find("main")
        if main is not None:
            text = main.get_text(separator=" ")
        else:
            text = soup.get_text(separator=" ")

        return clean_text(text)
    except Exception:
        return ""

def extract_text_from_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return clean_text(f.read())
    except Exception:
        return ""

def extract_text_from_html_or_mhtml(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        main = soup.find("main")
        if main is not None:
            text = main.get_text(separator=" ")
        else:
            text = soup.get_text(separator=" ")
        return clean_text(text)
    except Exception:
        return ""

def extract_text_from_pdf(path: str) -> str:
    try:
        import PyPDF2
        text_parts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
        return clean_text(" ".join(text_parts))
    except Exception:
        return ""

def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return extract_text_from_txt(path)
    if ext in [".html", ".htm", ".mhtml"]:
        return extract_text_from_html_or_mhtml(path)
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    return ""

def build_medical_documents(medical_sources: dict):
    documents = []

    for url in medical_sources.get("urls", []):
        text = extract_text_from_url(url)
        if text:
            documents.append({
                "source": url,
                "type": "url",
                "text": text
            })

    for path in medical_sources.get("pdf", []):
        text = extract_text_from_file(path)
        if text:
            documents.append({
                "source": path,
                "type": "file",
                "text": text
            })

    return documents
