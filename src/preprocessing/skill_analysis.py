import os
import re
from collections import Counter

import pandas as pd


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

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "eda"
)

SKILLS = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "natural language processing",
    "nlp",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "docker",
    "aws",
    "azure",
    "git",
    "javascript",
    "react",
    "fastapi",
    "flask",
    "excel",
    "communication",
    "leadership",
    "data analysis"
]


def count_skills(text):
    """Count known skills using phrase-aware matching."""

    text = str(text).lower()
    counts = Counter()

    for skill in SKILLS:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        matches = re.findall(pattern, text)

        if matches:
            counts[skill] += len(matches)

    return counts


def analyze_dataset(df, text_column, dataset_name):
    """Calculate skill frequency across records."""

    total_records = len(df)
    record_counts = Counter()
    mention_counts = Counter()

    for text in df[text_column].fillna(""):
        found = count_skills(text)

        # Count each skill at most once per record
        for skill, count in found.items():
            record_counts[skill] += 1
            mention_counts[skill] += count

    results = pd.DataFrame([
        {
            "skill": skill,
            "records_containing_skill": record_counts[skill],
            "total_mentions": mention_counts[skill],
            "percentage_of_records": (
                record_counts[skill] / total_records * 100
                if total_records else 0
            )
        }
        for skill in SKILLS
    ])

    results = results.sort_values(
        "records_containing_skill",
        ascending=False
    )

    print(f"\n{'=' * 60}")
    print(f"{dataset_name.upper()} SKILL ANALYSIS")
    print(f"{'=' * 60}")

    print(f"Total records analyzed: {total_records}")
    print("\nMost frequent skills:")
    print(results.head(15).to_string(index=False))

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{dataset_name}_skill_frequency.csv"
    )

    results.to_csv(output_path, index=False)

    print(f"\nSaved results: {output_path}")

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resumes = pd.read_csv(RESUME_PATH)
    internships = pd.read_csv(INTERNSHIP_PATH)

    resume_results = analyze_dataset(
        resumes,
        "Resume_str",
        "resume"
    )

    internship_results = analyze_dataset(
        internships,
        "description",
        "internship"
    )

    print("\nSkill analysis completed.")

    # Return values are available for future analysis.
    return resume_results, internship_results


if __name__ == "__main__":
    main()