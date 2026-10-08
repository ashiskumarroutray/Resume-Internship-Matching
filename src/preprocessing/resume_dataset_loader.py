import os
import pandas as pd


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


def load_resume_dataset():
    """
    Load the Resume.csv dataset.

    Expected columns:
    - ID
    - Resume_str
    - Resume_html
    - Category
    """

    csv_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "resumes",
        "Resume.csv"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Resume dataset not found at: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = ["ID", "Resume_str", "Category"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' not found in Resume.csv"
            )

    # Remove rows where resume text is missing
    df = df.dropna(subset=["Resume_str"])

    # Convert resume text to string
    df["Resume_str"] = df["Resume_str"].astype(str)

    # Reset index
    df = df.reset_index(drop=True)

    print(f"Resume dataset loaded successfully.")
    print(f"Total resumes: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


def get_resume_by_index(df, index=0):
    """
    Get one resume from the dataset.
    """

    if index < 0 or index >= len(df):
        raise IndexError(
            f"Resume index {index} is out of range. "
            f"Dataset contains {len(df)} resumes."
        )

    resume = df.iloc[index]

    return {
        "id": resume["ID"],
        "text": resume["Resume_str"],
        "category": resume["Category"]
    }


if __name__ == "__main__":

    df = load_resume_dataset()

    resume = get_resume_by_index(df, index=0)

    print("\n" + "=" * 70)
    print("SAMPLE RESUME")
    print("=" * 70)

    print(f"\nResume ID: {resume['id']}")
    print(f"Category: {resume['category']}")

    print("\nResume Text:")
    print(resume["text"][:2000])