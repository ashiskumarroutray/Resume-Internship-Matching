from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.matching.hybrid_matcher import (
    calculate_skill_score,
)

LABELS_PATH = ROOT / "data" / "processed" / "relevance_labels.csv"
RESUMES_PATH = ROOT / "data" / "processed" / "resumes_processed.csv"
JOBS_PATH = ROOT / "data" / "processed" / "internships_processed.csv"


def main():
    labels = pd.read_csv(LABELS_PATH)
    resumes = pd.read_csv(RESUMES_PATH)
    jobs = pd.read_csv(JOBS_PATH)

    print("Loaded relevance labels:", len(labels))
    print("Loaded resumes:", len(resumes))
    print("Loaded internships:", len(jobs))

    required_labels = {"resume_id", "job_id", "relevant"}
    if not required_labels.issubset(labels.columns):
        raise ValueError(
            f"Labels CSV must contain: {required_labels}"
        )

    # Adjust these mappings if your CSVs use different ID columns.
    resume_id_col = "ID"
    job_id_col = "job_id"

    if resume_id_col not in resumes.columns:
        raise ValueError(
            f"Resume dataset is missing '{resume_id_col}'. "
            f"Available columns: {list(resumes.columns)}"
        )

    if job_id_col not in jobs.columns:
        raise ValueError(
            f"Internship dataset is missing '{job_id_col}'. "
            f"Available columns: {list(jobs.columns)}"
        )

    resumes[resume_id_col] = resumes[resume_id_col].astype(str)
    jobs[job_id_col] = jobs[job_id_col].astype(str)

    # Relevance-label resume_id values are zero-based row indexes.
    resumes = resumes.reset_index(drop=True)
    resumes["_label_resume_id"] = resumes.index.astype(str)

    resume_lookup = resumes.set_index("_label_resume_id")
    job_lookup = jobs.copy()
    job_lookup[job_id_col] = job_lookup[job_id_col].astype(str)
    job_lookup = job_lookup.set_index(job_id_col)
    predictions = []
    actual = []

    for row in labels.itertuples(index=False):
        resume_id = str(row.resume_id)
        job_id = str(row.job_id)

        if resume_id not in resume_lookup.index:
            raise ValueError(
                f"Resume ID {resume_id} was not found. "
                "Check whether resume_id represents a dataset index "
                "or the actual resume ID."
            )

        if job_id not in job_lookup.index:
            raise ValueError(
                f"Job ID {job_id} was not found. "
                "Check whether job_id represents an index or actual ID."
            )

        resume_text = str(resume_lookup.loc[resume_id]["Resume_str"])
        job = job_lookup.loc[job_id]
        job_text = (
            str(job.get("title", ""))
            + " "
            + str(job.get("description", ""))
        )

        score, matched, missing = calculate_skill_score(
            resume_text, job_text
        )

        # Initial rule-based prediction threshold.
        # This is a baseline decision rule, not a calibrated threshold.
        print(
        f"Resume={resume_id}, Job={job_id}, "
        f"Skill score={score:.3f}, Actual={row.relevant}"
)

        predicted_relevant = int(score > 0)

        actual.append(int(row.relevant))
        predictions.append(predicted_relevant)

    print("\n===== RESUME MATCHING EVALUATION =====")
    print("Labeled pairs:", len(actual))
    print(f"Accuracy:  {accuracy_score(actual, predictions) * 100:.2f}%")
    print(
        f"Precision: {precision_score(actual, predictions, zero_division=0) * 100:.2f}%"
    )
    print(
        f"Recall:    {recall_score(actual, predictions, zero_division=0) * 100:.2f}%"
    )
    print(
        f"F1-score:  {f1_score(actual, predictions, zero_division=0) * 100:.2f}%"
    )

    print("\nClassification report:")
    print(
        classification_report(
            actual,
            predictions,
            target_names=["Not relevant", "Relevant"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()