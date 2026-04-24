import pdfplumber
import PyPDF2
import os
import csv
import re


# -----------------------------
# PDF EXTRACTION (pdfplumber)
# -----------------------------
def extract_text_pdfplumber(pdf_path):
    text_pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text(
                x_tolerance=2,
                y_tolerance=2
            )
            if extracted:
                text_pages.append(extracted)

    return "\n".join(text_pages)


# -----------------------------
# PDF EXTRACTION (PyPDF2 fallback)
# -----------------------------
def extract_text_pypdf2(pdf_path):
    text_pages = []

    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_pages.append(extracted)

    return "\n".join(text_pages)


# -----------------------------
# CLEANING (SAFE VERSION)
# -----------------------------
def clean_text(text):
    if not text:
        return ""

    # Normalize newlines first
    text = text.replace('\r', '\n')

    # Fix hyphenation across line breaks
    text = re.sub(r'-\n\s*', '', text)

    # Replace newlines with space (but keep structure first)
    text = text.replace('\n', ' ')

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove weird control characters
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)

    return text.strip()


# -----------------------------
# MAIN EXTRACTION FUNCTION
# -----------------------------
def extract_from_pdf(pdf_path):
    text = ""

    # Try pdfplumber first
    try:
        text = extract_text_pdfplumber(pdf_path)
        if text and len(text.strip()) > 100:
            return clean_text(text)
    except Exception as e:
        print(f"pdfplumber failed for {pdf_path}: {e}")

    # Fallback PyPDF2
    try:
        text = extract_text_pypdf2(pdf_path)
        return clean_text(text)
    except Exception as e:
        print(f"PyPDF2 failed for {pdf_path}: {e}")

    return ""


# -----------------------------
# PROCESS FOLDER
# -----------------------------
def process_pdf_folder(folder_path, label, output_csv):
    results = []

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]

    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(folder_path, filename)

        text = extract_from_pdf(pdf_path)

        if text and len(text.strip()) > 50:
            results.append({
                'filename': filename,
                'text': text,
                'label': label
            })
            print(f"Processed {i+1}/{len(pdf_files)}: {filename}")
        else:
            print(f"Failed / empty: {filename}")

    # -----------------------------
    # SAFE CSV WRITING
    # -----------------------------
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['filename', 'text', 'label'],
            quoting=csv.QUOTE_ALL,
            escapechar='\\'
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! {len(results)} papers saved to {output_csv}")


# -----------------------------
# RUN
# -----------------------------
process_pdf_folder(
    '../../data/real_papers',
    'Human',
    '../../data/real_papers.csv'
)
