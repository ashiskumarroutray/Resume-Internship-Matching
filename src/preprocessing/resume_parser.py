import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF resume.
    """
    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text.strip()


if __name__ == "__main__":
    pdf_path = input("Enter the path to the resume PDF: ")

    resume_text = extract_text_from_pdf(pdf_path)

    print("\n" + "=" * 60)
    print("EXTRACTED RESUME TEXT")
    print("=" * 60)

    print(resume_text)