import os
import re
import argparse

import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer

from src.matching.hybrid_matcher import extract_skills
from src.matching.profile_extractor import (
    extract_years_of_experience,
    extract_education,
)
from src.recommendation.learning_roadmap import (
    generate_learning_roadmap,
)


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

RESUME_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "resumes_processed.csv",
)

INTERNSHIP_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "internships_processed.csv",
)

MODEL_NAME = "all-MiniLM-L6-v2"

# Hybrid ranking weights. These are initial weights and
# should eventually be validated against labeled data.
WEIGHTS = {
    "semantic": 0.40,
    "skills": 0.30,
    "experience": 0.15,
    "education": 0.15,
}


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text):
    """Normalize whitespace without destroying technical terms."""

    if text is None or pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_skills(text):
    """Extract skills using the shared skill extractor."""

    return set(extract_skills(normalize_text(text)))


# ============================================================
# DATA LOADING
# ============================================================

def load_resumes():
    """Load and validate the processed resume dataset."""

    if not os.path.exists(RESUME_PATH):
        raise FileNotFoundError(
            f"Resume dataset not found: {RESUME_PATH}\n"
            "Run the data-cleaning pipeline first."
        )

    df = pd.read_csv(RESUME_PATH)

    required = ["ID", "Resume_str", "Category"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Resume dataset is missing columns: {sorted(missing)}"
        )

    df = df.dropna(subset=["Resume_str"]).copy()

    df["Resume_str"] = df["Resume_str"].apply(normalize_text)
    df["ID"] = df["ID"].astype(str)

    df = df[df["Resume_str"].str.len() > 0]

    return df.reset_index(drop=True)


def load_internships():
    """Load and validate the processed internship dataset."""

    if not os.path.exists(INTERNSHIP_PATH):
        raise FileNotFoundError(
            f"Internship dataset not found: {INTERNSHIP_PATH}\n"
            "Run the data-cleaning pipeline first."
        )

    df = pd.read_csv(INTERNSHIP_PATH)

    required = [
        "job_id",
        "title",
        "company",
        "description",
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            "Internship dataset is missing columns: "
            f"{sorted(missing)}"
        )

    df = df.dropna(subset=["description"]).copy()

    df["job_id"] = df["job_id"].astype(str)

    for column in ["title", "company", "description"]:
        df[column] = df[column].apply(normalize_text)

    df = df[df["description"].str.len() > 0]

    return df.reset_index(drop=True)


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def calculate_skill_score(resume_text, job_text):
    """
    Calculate required-skill coverage.

    Returns:
        score: float between 0 and 1
        matched_skills: list of matched skills
        missing_skills: list of missing skills
    """

    resume_skills = get_skills(resume_text)
    job_skills = get_skills(job_text)

    # No identifiable job skills means coverage is unknown.
    if not job_skills:
        return 0.5, [], []

    matched = resume_skills & job_skills
    missing = job_skills - resume_skills

    score = len(matched) / len(job_skills)

    return (
        float(score),
        sorted(matched),
        sorted(missing),
    )


def calculate_experience_score(resume_text, job_text):
    """
    Compare candidate experience with the experience requirement.

    An undetected job requirement receives a neutral score.
    """

    candidate_years = extract_years_of_experience(resume_text)
    required_years = extract_years_of_experience(job_text)

    if required_years <= 0:
        return 0.5

    if candidate_years >= required_years:
        return 1.0

    if candidate_years <= 0:
        return 0.0

    return min(candidate_years / required_years, 1.0)


def calculate_education_score(resume_text, job_text):
    """
    Compare detected education terms.

    If no education requirement is detected, use a neutral score.
    """

    candidate_education = set(
        extract_education(resume_text)
    )

    required_education = set(
        extract_education(job_text)
    )

    if not required_education:
        return 0.5

    if not candidate_education:
        return 0.0

    matched = candidate_education & required_education

    return len(matched) / len(required_education)


def calculate_hybrid_score(
    semantic_score,
    skill_score,
    experience_score,
    education_score,
):
    """
    Calculate a deterministic hybrid score from 0 to 100.

    Input scores must be between 0 and 1.
    """

    components = {
        "semantic": semantic_score,
        "skills": skill_score,
        "experience": experience_score,
        "education": education_score,
    }

    for name, value in components.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} score must be numeric.")

        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} score must be between 0 and 1; got {value}."
            )

    score = sum(
        WEIGHTS[name] * components[name]
        for name in WEIGHTS
    )

    return round(float(score) * 100, 2)


# ============================================================
# CORE MATCHING ENGINE
# ============================================================

class CoreMatcher:
    """
    Shared matching engine for student and recruiter workflows.

    Student mode:
        Resume -> ranked internships

    Recruiter mode:
        Job description -> ranked candidate resumes
    """

    def __init__(self):
        print(f"Loading Sentence Transformer: {MODEL_NAME}")

        self.model = SentenceTransformer(MODEL_NAME)

        self.index = None
        self.items = []
        self.item_texts = []
        self.mode = None

    def build_index(self, items, text_column, mode):
        """Build a normalized FAISS index for the chosen dataset."""

        if mode not in {"student", "recruiter"}:
            raise ValueError(
                "mode must be 'student' or 'recruiter'"
            )

        if not items:
            raise ValueError(
                "Cannot build an index from an empty dataset."
            )

        self.mode = mode
        self.items = items

        self.item_texts = [
            normalize_text(item.get(text_column, ""))
            for item in items
        ]

        if not any(self.item_texts):
            raise ValueError(
                f"No usable text found in column '{text_column}'."
            )

        print(f"Embedding {len(self.item_texts)} records...")

        embeddings = self.model.encode(
            self.item_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2 or embeddings.shape[0] != len(items):
            raise ValueError(
                "Unexpected embedding dimensions returned by the model."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain invalid numeric values."
            )

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        self.index.add(embeddings)

        print(
            f"FAISS index ready: {len(items)} records "
            f"({mode} mode)."
        )

    def rank(self, query_text, top_k=10):
        """
        Rank indexed records against the query.

        Returns dictionaries containing the original record,
        component scores, matched skills, and missing skills.
        """

        if self.index is None:
            raise RuntimeError(
                "Build the FAISS index before calling rank()."
            )

        query_text = normalize_text(query_text)

        if not query_text:
            raise ValueError("Query text cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if not np.isfinite(query_embedding).all():
            raise ValueError(
                "Query embedding contains invalid numeric values."
            )

        faiss.normalize_L2(query_embedding)

        # Search the entire index so all candidates can be reranked
        # using the final hybrid score.
        scores, indices = self.index.search(
            query_embedding,
            len(self.items),
        )

        results = []

        for similarity, item_index in zip(
            scores[0],
            indices[0],
        ):
            if item_index < 0:
                continue

            item_index = int(item_index)
            item = self.items[item_index]

            if self.mode == "student":
                resume_text = query_text
                job_text = self.item_texts[item_index]
            else:
                resume_text = self.item_texts[item_index]
                job_text = query_text

            # Cosine similarity is converted to a bounded score.
            semantic_score = float(
                np.clip(similarity, 0.0, 1.0)
            )

            skill_score, matched, missing = (
                calculate_skill_score(
                    resume_text,
                    job_text,
                )
            )

            experience_score = calculate_experience_score(
                resume_text,
                job_text,
            )

            education_score = calculate_education_score(
                resume_text,
                job_text,
            )

            final_score = calculate_hybrid_score(
                semantic_score,
                skill_score,
                experience_score,
                education_score,
            )

            result = item.copy()

            result.update({
                "semantic_score": round(
                    semantic_score * 100, 2
                ),
                "skill_score": round(
                    skill_score * 100, 2
                ),
                "experience_score": round(
                    experience_score * 100, 2
                ),
                "education_score": round(
                    education_score * 100, 2
                ),
                "final_score": final_score,
                "matched_skills": matched,
                "missing_skills": missing,
            })

            results.append(result)

        # The hybrid score, not the FAISS score alone,
        # determines the final order.
        results.sort(
            key=lambda row: row["final_score"],
            reverse=True,
        )

        return results[:min(top_k, len(results))]


# ============================================================
# LEARNING ROADMAP INTEGRATION
# ============================================================

def attach_learning_roadmaps(
    results,
    mode,
    selected_resume=None,
    selected_job=None,
):
    """
    Attach a learning roadmap to each matching result.

    Student mode:
        Selected resume + each recommended internship.

    Recruiter mode:
        Each ranked resume + selected job description.

    This function reuses the existing roadmap generator.
    """

    if mode not in {"student", "recruiter"}:
        raise ValueError(
            "mode must be 'student' or 'recruiter'"
        )

    if mode == "student" and selected_resume is None:
        raise ValueError(
            "selected_resume is required in student mode."
        )

    if mode == "recruiter" and selected_job is None:
        raise ValueError(
            "selected_job is required in recruiter mode."
        )

    for result in results:

        if mode == "student":
            resume_text = normalize_text(
                selected_resume.get("Resume_str", "")
            )

            job_text = " ".join([
                normalize_text(result.get("title", "")),
                normalize_text(result.get("description", "")),
            ])

        else:
            resume_text = normalize_text(
                result.get("Resume_str", "")
            )

            job_text = " ".join([
                normalize_text(selected_job.get("title", "")),
                normalize_text(selected_job.get("description", "")),
            ])

        try:
            result["learning_roadmap"] = generate_learning_roadmap(
                resume_text,
                job_text,
            )
        except Exception as error:
            # A roadmap error should not discard an otherwise valid match.
            result["learning_roadmap"] = {
                "matched_skills": result.get("matched_skills", []),
                "missing_skills": result.get("missing_skills", []),
                "roadmap": [],
                "message": (
                    "A roadmap could not be generated: "
                    f"{error}"
                ),
            }

    return results


# ============================================================
# RESULT DISPLAY
# ============================================================

def display_results(results, mode):
    """Display ranked matches, score explanations, and roadmaps."""

    print("\n" + "=" * 75)

    if mode == "student":
        print("STUDENT MODE: BEST-MATCHING INTERNSHIPS")
    else:
        print("RECRUITER MODE: BEST-MATCHING CANDIDATES")

    print("=" * 75)

    if not results:
        print("No matching results were found.")
        return

    for rank, result in enumerate(results, start=1):

        print(f"\n#{rank}")

        if mode == "student":
            print(
                "Internship:",
                result.get("title", "Unknown"),
            )
            print(
                "Company:",
                result.get("company", "Unknown"),
            )
            print(
                "Job ID:",
                result.get("job_id", "Unknown"),
            )
        else:
            print(
                "Candidate ID:",
                result.get("ID", "Unknown"),
            )
            print(
                "Category:",
                result.get("Category", "Unknown"),
            )

        print(
            f"Final match score: "
            f"{result['final_score']:.2f}%"
        )

        print(
            f"Semantic score: "
            f"{result['semantic_score']:.2f}%"
        )

        print(
            f"Skill score: "
            f"{result['skill_score']:.2f}%"
        )

        print(
            f"Experience score: "
            f"{result['experience_score']:.2f}%"
        )

        print(
            f"Education score: "
            f"{result['education_score']:.2f}%"
        )

        print(
            "Matched skills:",
            result.get("matched_skills", []),
        )

        print(
            "Missing skills:",
            result.get("missing_skills", []),
        )

        # Display the personalized roadmap attached to this result.
        roadmap_data = result.get("learning_roadmap", {})
        roadmap = roadmap_data.get("roadmap", [])

        print("\nLearning roadmap:")

        if roadmap:
            for step_number, step in enumerate(roadmap, start=1):
                print(
                    f"  {step_number}. {step.get('skill', 'Unknown')} "
                    f"({step.get('priority', 'Normal')} priority)"
                )

                print(
                    f"     Reason: {step.get('reason', '')}"
                )

                print(
                    f"     Resource: {step.get('resource', '')}"
                )
        else:
            print(
                "  No additional recommendations were generated."
            )

        if roadmap_data.get("message"):
            print("  Note:", roadmap_data["message"])


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Two-sided AI resume–internship matching engine."
    )

    parser.add_argument(
        "--mode",
        choices=["student", "recruiter"],
        required=True,
        help=(
            "student: rank internships for a resume; "
            "recruiter: rank resumes for a job description"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to display.",
    )

    parser.add_argument(
        "--resume-index",
        type=int,
        default=0,
        help="Zero-based resume row index for student mode.",
    )

    parser.add_argument(
        "--job-index",
        type=int,
        default=0,
        help="Zero-based internship row index for recruiter mode.",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero.")

    resumes_df = load_resumes()
    internships_df = load_internships()

    if resumes_df.empty:
        raise ValueError("The processed resume dataset is empty.")

    if internships_df.empty:
        raise ValueError("The processed internship dataset is empty.")

    resumes = resumes_df.to_dict("records")
    internships = internships_df.to_dict("records")

    matcher = CoreMatcher()

    if args.mode == "student":

        if not 0 <= args.resume_index < len(resumes):
            raise IndexError(
                f"Resume index {args.resume_index} is outside "
                f"the dataset range 0–{len(resumes) - 1}."
            )

        selected_resume = resumes[args.resume_index]

        matcher.build_index(
            internships,
            text_column="description",
            mode="student",
        )

        results = matcher.rank(
            selected_resume["Resume_str"],
            top_k=args.top_k,
        )

        # Attach a separate roadmap for each recommended internship.
        results = attach_learning_roadmaps(
            results=results,
            mode="student",
            selected_resume=selected_resume,
        )

    else:

        if not 0 <= args.job_index < len(internships):
            raise IndexError(
                f"Internship index {args.job_index} is outside "
                f"the dataset range 0–{len(internships) - 1}."
            )

        selected_job = internships[args.job_index]

        matcher.build_index(
            resumes,
            text_column="Resume_str",
            mode="recruiter",
        )

        job_text = " ".join([
            selected_job.get("title", ""),
            selected_job.get("description", ""),
        ])

        results = matcher.rank(
            job_text,
            top_k=args.top_k,
        )

        # Attach a roadmap for each candidate relative to this job.
        results = attach_learning_roadmaps(
            results=results,
            mode="recruiter",
            selected_job=selected_job,
        )

    display_results(results, args.mode)


if __name__ == "__main__":
    main()