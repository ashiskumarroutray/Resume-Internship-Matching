import os
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

RESUME_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "resumes_processed.csv"
)

INTERNSHIP_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "internships_processed.csv"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "eda"
)


def resume_eda():

    print("\n" + "=" * 70)
    print("RESUME DATASET EDA")
    print("=" * 70)

    df = pd.read_csv(RESUME_PATH)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nCategory distribution:")
    print(df["Category"].value_counts())

    # Resume length
    df["resume_length"] = df["Resume_str"].str.len()

    print("\nResume length statistics:")
    print(df["resume_length"].describe())

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Category distribution
    plt.figure(figsize=(12, 6))

    df["Category"].value_counts().head(15).plot(
        kind="bar"
    )

    plt.title("Top Resume Categories")
    plt.xlabel("Category")
    plt.ylabel("Number of Resumes")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "resume_category_distribution.png"
        )
    )

    plt.close()

    # Resume length distribution
    plt.figure(figsize=(10, 6))

    df["resume_length"].plot(
        kind="hist",
        bins=30
    )

    plt.title("Resume Length Distribution")
    plt.xlabel("Characters")
    plt.ylabel("Number of Resumes")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "resume_length_distribution.png"
        )
    )

    plt.close()

    return df


def internship_eda():

    print("\n" + "=" * 70)
    print("INTERNSHIP DATASET EDA")
    print("=" * 70)

    df = pd.read_csv(INTERNSHIP_PATH)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nInternship titles:")
    print(df["title"].value_counts())

    print("\nCompanies:")
    print(df["company"].value_counts())

    df["description_length"] = (
        df["description"].str.len()
    )

    print("\nDescription length statistics:")
    print(df["description_length"].describe())

    return df


def main():

    resume_df = resume_eda()

    internship_df = internship_eda()

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)

    print(
        f"\nResume records analyzed: {len(resume_df)}"
    )

    print(
        f"Internship records analyzed: {len(internship_df)}"
    )

    print(
        f"\nEDA charts saved to:\n{RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()