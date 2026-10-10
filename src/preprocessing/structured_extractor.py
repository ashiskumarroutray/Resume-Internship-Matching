import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.matching.hybrid_matcher import extract_skills
from src.matching.profile_extractor import (
    extract_years_of_experience,
    extract_education,
)


def extract_structured_fields(text):
    """Extract skills, experience, and education from text."""

    text = str(text)

    skills = extract_skills(text)
    experience = extract_years_of_experience(text)
    education = extract_education(text)

    return {
        "skills": skills,
        "experience": experience,
        "education": education,
    }


def preprocess_resume_records():
    path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "resumes_processed.csv",
    )

    df = pd.read_csv(path)

    records = []

    for _, row in df.iterrows():
        fields = extract_structured_fields(row["Resume_str"])

        records.append({
            "ID": row["ID"],
            "Category": row["Category"],
            "Resume_str": row["Resume_str"],
            "skills": ", ".join(fields["skills"]),
            "experience": fields["experience"],
            "education": ", ".join(fields["education"]),
        })

    result = pd.DataFrame(records)

    output_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "structured_resumes.csv",
    )

    result.to_csv(output_path, index=False)

    print(f"\nStructured resumes saved: {output_path}")
    print(f"Records processed: {len(result)}")

    return result


def preprocess_internship_records():
    path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "internships_processed.csv",
    )

    df = pd.read_csv(path)

    records = []

    for _, row in df.iterrows():
        fields = extract_structured_fields(row["description"])

        records.append({
            "job_id": row["job_id"],
            "title": row["title"],
            "company": row["company"],
            "description": row["description"],
            "required_skills": ", ".join(fields["skills"]),
            "experience": fields["experience"],
            "education": ", ".join(fields["education"]),
        })

    result = pd.DataFrame(records)

    output_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "structured_internships.csv",
    )

    result.to_csv(output_path, index=False)

    print(f"\nStructured internships saved: {output_path}")
    print(f"Records processed: {len(result)}")

    return result


def main():
    print("\n" + "=" * 60)
    print("STRUCTURED DATA EXTRACTION")
    print("=" * 60)

    resumes = preprocess_resume_records()
    internships = preprocess_internship_records()

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Total resumes: {len(resumes)}")
    print(f"Total internships: {len(internships)}")


if __name__ == "__main__":
    main()