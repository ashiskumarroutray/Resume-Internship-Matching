from sentence_transformers import SentenceTransformer
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.resume_parser import extract_text_from_pdf


model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embedding(text):
    return model.encode(text)


if __name__ == "__main__":

    pdf_path = "data/raw/resumes/sample_resume.pdf"

    resume_text = extract_text_from_pdf(pdf_path)

    embedding = generate_embedding(resume_text)

    print("\nResume embedding generated successfully!")
    print("Resume text length:", len(resume_text))
    print("Embedding dimensions:", embedding.shape)
    print("First 10 values:", embedding[:10])