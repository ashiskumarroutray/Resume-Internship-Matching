import re

from src.matching.skill_normalizer import normalize_skill, normalize_skills


# ============================================================
# SKILL VOCABULARY
# ============================================================

SKILL_VOCABULARY = [
    # Programming languages
    "python",
    "java",
    "javascript",
    "typescript",
    "c",
    "c++",
    "c#",
    "sql",
    "r",
    "go",
    "rust",
    "php",
    "ruby",
    "swift",
    "kotlin",

    # Web development
    "html",
    "css",
    "react",
    "angular",
    "vue",
    "node.js",
    "express",
    "django",
    "flask",
    "fastapi",
    "spring boot",

    # Data and databases
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "power bi",
    "tableau",
    "excel",
    "mysql",
    "postgresql",
    "mongodb",
    "redis",
    "sqlite",

    # AI and machine learning
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "generative ai",
    "large language models",
    "llm",
    "neural networks",
    "reinforcement learning",
    "feature engineering",
    "model evaluation",
    "data analysis",
    "data science",
    "statistics",

    # Machine learning frameworks
    "tensorflow",
    "pytorch",
    "keras",
    "xgboost",
    "lightgbm",
    "opencv",
    "hugging face",
    "transformers",
    "langchain",
    "faiss",

    # Cloud, tools, and engineering
    "git",
    "github",
    "docker",
    "kubernetes",
    "linux",
    "aws",
    "azure",
    "google cloud",
    "gcp",
    "rest api",
    "graphql",
    "ci/cd",
    "airflow",
    "spark",
    "hadoop",
    "mlflow",
]


# ============================================================
# SKILL ALIASES
# ============================================================

# These aliases help detect alternative ways of writing a skill.
# Canonical names are defined in skill_normalizer.py.

DETECTION_ALIASES = {
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "ai": "artificial intelligence",
    "genai": "generative ai",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "tf": "tensorflow",
    "js": "javascript",
    "postgres": "postgresql",
    "postgres sql": "postgresql",
    "node": "node.js",
    "llms": "large language models",
}


# ============================================================
# SKILL EXTRACTION
# ============================================================

def _contains_phrase(text, phrase):
    """
    Check whether a phrase occurs as a complete token sequence.

    This prevents 'r' from matching every letter r in a word
    and prevents 'java' from matching 'javascript'.
    """

    pattern = (
        r"(?<!\w)"
        + re.escape(phrase)
        + r"(?!\w)"
    )

    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_skills(text):
    """
    Extract recognized skills and return canonical skill names.

    Args:
        text: Resume or internship/job-description text.

    Returns:
        Sorted list of normalized skills.
    """

    if not isinstance(text, str) or not text.strip():
        return []

    text = text.lower()

    detected_skills = []

    # Detect canonical skills.
    for skill in SKILL_VOCABULARY:
        if _contains_phrase(text, skill):
            detected_skills.append(skill)

    # Detect aliases and convert them to canonical names.
    for alias, canonical_name in DETECTION_ALIASES.items():
        if _contains_phrase(text, alias):
            detected_skills.append(canonical_name)

    # Normalize and remove duplicates.
    return normalize_skills(detected_skills)


# ============================================================
# SKILL MATCHING
# ============================================================

def calculate_skill_score(resume_text, job_text):
    """
    Calculate required-skill coverage.

    Score = matched required skills / detected required skills.

    Returns:
        score: float between 0 and 1
        matched_skills: sorted list
        missing_skills: sorted list
    """

    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))

    # If no skills are detected in the job description,
    # a skill-match score cannot be established.
    if not job_skills:
        return 0.5, [], []

    matched_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills.difference(resume_skills)

    score = len(matched_skills) / len(job_skills)

    return (
        float(score),
        sorted(matched_skills),
        sorted(missing_skills),
    )


# ============================================================
# HYBRID SCORE
# ============================================================

def calculate_final_score(
    semantic_score,
    skill_score,
    experience_score=0.5,
    education_score=0.5,
):
    """
    Calculate the final hybrid matching score.

    All input scores must be on a 0-to-1 scale.

    Weights:
        Semantic similarity: 40%
        Skill coverage:      30%
        Experience:          15%
        Education:           15%

    Returns:
        Final score between 0 and 100.
    """

    weights = {
        "semantic": 0.40,
        "skills": 0.30,
        "experience": 0.15,
        "education": 0.15,
    }

    components = {
        "semantic": semantic_score,
        "skills": skill_score,
        "experience": experience_score,
        "education": education_score,
    }

    for name, value in components.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} score must be numeric.")

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} score must be between 0 and 1; got {value}."
            )

    final_score = sum(
        weights[name] * components[name]
        for name in weights
    )

    return round(final_score * 100, 2)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    resume = """
    Python developer with experience in machine learning,
    pandas, NumPy, scikit-learn, NLP, and TensorFlow.
    """

    job_description = """
    We need an intern skilled in Python, ML, NLP,
    TensorFlow, PyTorch, SQL, and Docker.
    """

    print("=" * 55)
    print("SKILL EXTRACTION TEST")
    print("=" * 55)

    print("\nResume skills:")
    print(extract_skills(resume))

    print("\nJob skills:")
    print(extract_skills(job_description))

    score, matched, missing = calculate_skill_score(
        resume,
        job_description,
    )

    print("\nMatched skills:")
    print(matched)

    print("\nMissing skills:")
    print(missing)

    print(f"\nSkill coverage: {score * 100:.2f}%")

    final_score = calculate_final_score(
        semantic_score=0.82,
        skill_score=score,
        experience_score=0.70,
        education_score=0.80,
    )

    print(f"Example hybrid score: {final_score:.2f}%")