from src.matching.skill_normalizer import normalize_skills


def extract_skills(text):
    skills = [
        "python", "java", "c++", "sql",
        "machine learning", "ml",
        "deep learning", "dl",
        "nlp", "natural language processing",
        "tensorflow", "pytorch",
        "scikit-learn", "sklearn",
        "pandas", "numpy",
        "docker", "aws", "azure", "gcp",
        "git", "github",
        "faiss", "langchain", "rag",
        "generative ai", "genai",
        "html", "css", "javascript",
        "react", "fastapi", "flask"
    ]

    text = text.lower()

    found_skills = []

    for skill in skills:
        if skill in text:
            found_skills.append(skill)

    # Convert aliases to standard names
    return normalize_skills(found_skills)


def calculate_skill_score(resume_text, job_text):

    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))

    if not job_skills:
        return 0, [], []

    matched_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills - resume_skills

    score = len(matched_skills) / len(job_skills)

    return (
        score,
        list(matched_skills),
        list(missing_skills)
    )


def calculate_final_score(
    semantic_score,
    skill_score,
    experience_score=0.0,
    education_score=0.0
):

    final_score = (
        0.40 * semantic_score
        + 0.30 * skill_score
        + 0.15 * experience_score
        + 0.15 * education_score
    )

    return final_score


if __name__ == "__main__":

    resume = """
    Python machine learning NLP TensorFlow
    Pandas NumPy SQL FAISS sklearn
    """

    internship = """
    Looking for a Machine Learning intern
    with Python, Natural Language Processing,
    TensorFlow, SQL, Docker and AWS experience.
    """

    skill_score, matched, missing = calculate_skill_score(
        resume,
        internship
    )

    final_score = calculate_final_score(
        semantic_score=0.80,
        skill_score=skill_score,
        experience_score=0.70,
        education_score=1.0
    )

    print("Skill Score:", round(skill_score, 3))
    print("Matched Skills:", matched)
    print("Missing Skills:", missing)
    print("Final Score:", round(final_score * 100, 2), "%")