import os
import re
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

RAW_RESUME_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "resumes",
    "Resume.csv"
)

RAW_INTERNSHIP_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "internships",
    "internships.csv"
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)


def clean_text(text):
    """
    Clean and normalize text while preserving useful
    technical terms such as C++, C#, .NET, etc.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert HTML-like tags to spaces
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize line breaks and tabs
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_resumes():
    print("\n" + "=" * 70)
    print("PREPROCESSING RESUME DATASET")
    print("=" * 70)

    df = pd.read_csv(RAW_RESUME_PATH)

    print(f"Original rows: {len(df)}")

    required_columns = [
        "ID",
        "Resume_str",
        "Category"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' not found in Resume.csv"
            )

    # Remove rows without resume text
    df = df.dropna(subset=["Resume_str"])

    # Convert resume text to string
    df["Resume_str"] = df["Resume_str"].astype(str)

    # Clean resume text
    df["Resume_str"] = df["Resume_str"].apply(clean_text)

    # Remove empty resumes
    df = df[df["Resume_str"].str.len() > 0]

    # Remove duplicate resume IDs
    df = df.drop_duplicates(subset=["ID"])

    # Remove exact duplicate resume text
    df = df.drop_duplicates(subset=["Resume_str"])

    # Clean category
    df["Category"] = df["Category"].fillna("Unknown")
    df["Category"] = df["Category"].astype(str).str.strip()

    df = df.reset_index(drop=True)

    output_path = os.path.join(
        PROCESSED_DIR,
        "resumes_processed.csv"
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Processed rows: {len(df)}")
    print(f"Saved to: {output_path}")

    return df


def preprocess_internships():
    print("\n" + "=" * 70)
    print("PREPROCESSING INTERNSHIP DATASET")
    print("=" * 70)

    df = pd.read_csv(RAW_INTERNSHIP_PATH)

    print(f"Original rows: {len(df)}")

    required_columns = [
        "job_id",
        "title",
        "company",
        "description"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' not found in internships.csv"
            )

    # Remove internships without descriptions
    df = df.dropna(subset=["description"])

    # Clean text fields
    for column in ["title", "company", "description"]:
        df[column] = df[column].fillna("")
        df[column] = df[column].apply(clean_text)

    # Remove internships with empty descriptions
    df = df[df["description"].str.len() > 0]

    # Remove duplicate job IDs
    df = df.drop_duplicates(subset=["job_id"])

    # Remove exact duplicate descriptions
    df = df.drop_duplicates(subset=["description"])

    df = df.reset_index(drop=True)

    output_path = os.path.join(
        PROCESSED_DIR,
        "internships_processed.csv"
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Processed rows: {len(df)}")
    print(f"Saved to: {output_path}")

    return df


def main():
    resumes = preprocess_resumes()
    internships = preprocess_internships()

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(f"\nFinal resume records: {len(resumes)}")
    print(f"Final internship records: {len(internships)}")


if __name__ == "__main__":
    main()