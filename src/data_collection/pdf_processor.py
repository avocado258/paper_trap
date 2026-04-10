import pdfplumber
import PyPDF2
import os
import csv

def extract_text_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def extract_text_pypdf2(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def clean_text(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        # Remove page numbers
        if line.isdigit():
            continue
        # Remove very short lines
        if len(line) < 10:
            continue
        cleaned.append(line)
    return ' '.join(cleaned)

def extract_from_pdf(pdf_path):
    # Try pdfplumber first
    try:
        text = extract_text_pdfplumber(pdf_path)
        if len(text) > 100:
            return clean_text(text)
    except:
        pass
    
    # Fallback to PyPDF2
    try:
        text = extract_text_pypdf2(pdf_path)
        return clean_text(text)
    except:
        return ""

# Process all PDFs
def process_pdf_folder(folder_path, label, output_csv):
    results = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(folder_path, filename)
        text = extract_from_pdf(pdf_path)
        
        if text:
            results.append({
                'filename': filename,
                'text': text,
                'label': label
            })
            print(f"Processed {i+1}/{len(pdf_files)}: {filename}")
        else:
            print(f"Failed: {filename}")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'text', 'label'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nDone! {len(results)} papers saved to {output_csv}")

process_pdf_folder('../../data/real_papers', 'Human', '../../data/real_papers.csv')