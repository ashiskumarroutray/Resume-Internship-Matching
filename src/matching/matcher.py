import os
import pandas as pd

from src.preprocessing.resume_parser import extract_text_from_pdf
from src.retrieval.faiss_search import InternshipSearch


# Project root directory
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


def main():

    # -----------------------------
    # 1. Load resume
    # -----------------------------

    resume_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "resumes",
        "sample_resume.pdf"
    )

    resume_text = extract_text_from_pdf(resume_path)

    print("\nResume loaded successfully.")
    print("Resume characters:", len(resume_text))


    # -----------------------------
    # 2. Load internship dataset
    # -----------------------------

    csv_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "internships",
        "internships.csv"
    )

    df = pd.read_csv(csv_path)

    internships = df.to_dict("records")

    print(f"Internships loaded: {len(internships)}")


    # -----------------------------
    # 3. Build FAISS index
    # -----------------------------

    search_engine = InternshipSearch()

    search_engine.build_index(internships)


    # -----------------------------
    # 4. Search for matching internships
    # -----------------------------

    results = search_engine.search(
        resume_text,
        top_k=5
    )


    # -----------------------------
    # 5. Display results
    # -----------------------------

    print("\n" + "=" * 70)
    print("TOP INTERNSHIP MATCHES")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        print(f"\n#{rank} - {result['title']}")
        print(f"Company: {result['company']}")
        print(
            f"Similarity Score: "
            f"{result['similarity_score']:.4f}"
        )
        print(f"Description: {result['description']}")


if __name__ == "__main__":
    main()