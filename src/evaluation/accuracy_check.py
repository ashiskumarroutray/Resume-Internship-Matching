from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.matching.core_matcher import (
    MODEL_NAME,
    normalize_text,
    calculate_skill_score,
    calculate_experience_score,
    calculate_education_score,
    calculate_hybrid_score,
)

from sentence_transformers import SentenceTransformer


# Dataset paths
RESUMES_PATH = ROOT / "data" / "processed" / "resumes_processed.csv"
JOBS_PATH = ROOT / "data" / "processed" / "internships_processed.csv"

# Support either existing location for the relevance labels.
LABEL_PATHS = [
    ROOT / "data" / "processed" / "relevance_labels.csv",
    ROOT / "src" / "evaluation" / "relevance_labels.csv",
]

OUTPUT_PATH = ROOT / "results" / "accuracy_predictions.csv"

# Explicit baseline threshold on the hybrid score (0-100).
# This is a starting decision rule, not a calibrated threshold.
THRESHOLD = 50.0


def main():
    labels_path = next(
        (path for path in LABEL_PATHS if path.exists()),
        None,
    )

    if labels_path is None:
        raise FileNotFoundError(
            "Could not find relevance_labels.csv. Checked:\n"
            + "\n".join(str(path) for path in LABEL_PATHS)
        )

    labels = pd.read_csv(labels_path)
    resumes = pd.read_csv(RESUMES_PATH).reset_index(drop=True)
    jobs = pd.read_csv(JOBS_PATH).reset_index(drop=True)

    required = {"resume_id", "job_id", "relevant"}
    if not required.issubset(labels.columns):
        raise ValueError(
            f"Labels must contain these columns: {sorted(required)}"
        )

    if not {"Resume_str", "ID"}.issubset(resumes.columns):
        raise ValueError("Resume dataset needs ID and Resume_str columns.")

    if not {"job_id", "title", "description"}.issubset(jobs.columns):
        raise ValueError(
            "Internship dataset needs job_id, title, and description."
        )

    if labels.empty:
        raise ValueError("The relevance-label dataset is empty.")

    if labels["relevant"].isna().any():
        raise ValueError("Relevance labels contain missing values.")

    labels["relevant"] = pd.to_numeric(
        labels["relevant"], errors="raise"
    ).astype(int)

    if not set(labels["relevant"].unique()).issubset({0, 1}):
        raise ValueError("The relevant column must contain only 0 and 1.")

    if labels.duplicated(["resume_id", "job_id"]).any():
        raise ValueError("Duplicate resume/job pairs found in labels.")

    # In this dataset, resume_id means the zero-based row index.
    resumes["_label_resume_id"] = resumes.index.astype(str)
    resumes["_label_resume_id"] = resumes["_label_resume_id"].str.strip()

    resumes["ID"] = resumes["ID"].astype(str).str.strip()
    jobs["job_id"] = jobs["job_id"].astype(str).str.strip()

    resume_lookup = resumes.set_index("_label_resume_id")
    job_lookup = jobs.set_index("job_id")

    # Resolve every label before loading the embedding model.
    pairs = []

    for row in labels.itertuples(index=False):
        resume_id = str(row.resume_id).strip()
        job_id = str(row.job_id).strip()

        if resume_id not in resume_lookup.index:
            raise ValueError(
                f"Resume row index {resume_id} is not in the dataset."
            )

        if job_id not in job_lookup.index:
            raise ValueError(
                f"Job ID {job_id} is not in the internship dataset."
            )

        resume_text = normalize_text(
            resume_lookup.loc[resume_id, "Resume_str"]
        )

        job = job_lookup.loc[job_id]

        # Use the title and description together for pair evaluation.
        job_text = normalize_text(
            f"{job['title']} {job['description']}"
        )

        if not resume_text or not job_text:
            raise ValueError(
                f"Empty text for resume {resume_id}, job {job_id}."
            )

        pairs.append({
            "resume_id": resume_id,
            "job_id": job_id,
            "resume_text": resume_text,
            "job_text": job_text,
            "actual": int(row.relevant),
        })

    print("Labels file:", labels_path)
    print("Labeled pairs:", len(pairs))
    print("Resumes available:", len(resumes))
    print("Internships available:", len(jobs))
    print("\nLoading semantic model:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    # Encode all texts in batches. Normalized embeddings allow
    # cosine similarity to be calculated using a dot product.
    resume_embeddings = model.encode(
        [pair["resume_text"] for pair in pairs],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    job_embeddings = model.encode(
        [pair["job_text"] for pair in pairs],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    results = []

    for index, pair in enumerate(pairs):
        # Semantic similarity
        semantic_score = float(
            np.clip(
                np.dot(
                    resume_embeddings[index],
                    job_embeddings[index],
                ),
                0.0,
                1.0,
            )
        )

        # The same component-scoring functions used by CoreMatcher.
        skill_score, matched, missing = calculate_skill_score(
            pair["resume_text"],
            pair["job_text"],
        )

        experience_score = calculate_experience_score(
            pair["resume_text"],
            pair["job_text"],
        )

        education_score = calculate_education_score(
            pair["resume_text"],
            pair["job_text"],
        )

        # Actual hybrid scoring formula: 40/30/15/15.
        final_score = calculate_hybrid_score(
            semantic_score,
            skill_score,
            experience_score,
            education_score,
        )

        predicted = int(final_score >= THRESHOLD)

        results.append({
            "resume_id": pair["resume_id"],
            "job_id": pair["job_id"],
            "actual": pair["actual"],
            "semantic_score": round(semantic_score * 100, 2),
            "skill_score": round(skill_score * 100, 2),
            "experience_score": round(experience_score * 100, 2),
            "education_score": round(education_score * 100, 2),
            "hybrid_score": final_score,
            "predicted": predicted,
            "matched_skills": ", ".join(matched),
            "missing_skills": ", ".join(missing),
        })

    report = pd.DataFrame(results)

    y_true = report["actual"].to_numpy()
    y_pred = report["predicted"].to_numpy()

    print("\n===== ACTUAL HYBRID MODEL EVALUATION =====")
    print("Pairs evaluated:", len(report))
    print("Threshold:", THRESHOLD, "/ 100")
    print("Actual relevant:", int((y_true == 1).sum()))
    print("Actual not relevant:", int((y_true == 0).sum()))
    print("Predicted relevant:", int((y_pred == 1).sum()))
    print("Predicted not relevant:", int((y_pred == 0).sum()))

    print(f"\nAccuracy:  {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(
        f"Precision: "
        f"{precision_score(y_true, y_pred, zero_division=0) * 100:.2f}%"
    )
    print(
        f"Recall:    "
        f"{recall_score(y_true, y_pred, zero_division=0) * 100:.2f}%"
    )
    print(
        f"F1-score:  "
        f"{f1_score(y_true, y_pred, zero_division=0) * 100:.2f}%"
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    print("\nConfusion matrix:")
    print(f"True negatives:  {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives:  {tp}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Not relevant", "Relevant"],
            zero_division=0,
        )
    )

    # ROC-AUC evaluates score ordering independently of the
    # selected binary threshold. It requires both label classes.
    if len(np.unique(y_true)) == 2:
        print(
            "ROC-AUC:   "
            f"{roc_auc_score(y_true, report['hybrid_score']) * 100:.2f}%"
        )
    else:
        print("ROC-AUC unavailable: labels contain only one class.")

    print("\n===== PER-PAIR RESULTS =====")
    print(
        report[
            [
                "resume_id",
                "job_id",
                "actual",
                "hybrid_score",
                "predicted",
            ]
        ].to_string(index=False)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print("\nDetailed results saved to:", OUTPUT_PATH)
    print(
        "\nLIMITATION: This is a preliminary evaluation on a small "
        "labeled dataset. The 50-point threshold is a baseline, "
        "not a validated or calibrated decision boundary."
    )


if __name__ == "__main__":
    main()