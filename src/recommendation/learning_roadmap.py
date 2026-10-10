from src.matching.hybrid_matcher import extract_skills


# Prerequisite relationships between skills.
PREREQUISITES = {
    "machine learning": ["python", "statistics"],
    "deep learning": ["python", "machine learning"],
    "natural language processing": ["python", "machine learning"],
    "computer vision": ["python", "deep learning"],
    "generative ai": ["python", "natural language processing"],
    "tensorflow": ["python", "machine learning"],
    "pytorch": ["python", "machine learning"],
    "scikit-learn": ["python"],
    "pandas": ["python"],
    "numpy": ["python"],
    "data science": ["python", "statistics"],
    "data analysis": ["python", "excel"],
    "power bi": ["excel", "sql"],
    "tableau": ["excel", "sql"],
    "fastapi": ["python", "rest api"],
    "django": ["python"],
    "react": ["javascript", "html", "css"],
    "node.js": ["javascript"],
    "docker": ["linux"],
    "kubernetes": ["docker"],
}


# Suggested learning resources.
# These are general starting points, not endorsements.
LEARNING_RESOURCES = {
    "python": "https://docs.python.org/3/tutorial/",
    "machine learning": "https://scikit-learn.org/stable/user_guide.html",
    "deep learning": "https://www.deeplearning.ai/",
    "natural language processing": "https://huggingface.co/learn/nlp-course/",
    "generative ai": "https://huggingface.co/learn",
    "tensorflow": "https://www.tensorflow.org/tutorials",
    "pytorch": "https://pytorch.org/tutorials/",
    "scikit-learn": "https://scikit-learn.org/stable/tutorial/index.html",
    "pandas": "https://pandas.pydata.org/docs/getting_started/",
    "numpy": "https://numpy.org/learn/",
    "sql": "https://sqlbolt.com/",
    "javascript": "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
    "react": "https://react.dev/learn",
    "docker": "https://docs.docker.com/get-started/",
    "kubernetes": "https://kubernetes.io/docs/tutorials/",
    "statistics": "https://www.khanacademy.org/math/statistics-probability",
}


def generate_learning_roadmap(resume_text, job_description):
    """
    Generate a basic learning roadmap from missing job skills.

    The roadmap is rule-based and uses the existing skill extractor.
    """

    resume_skills = set(extract_skills(resume_text))
    required_skills = set(extract_skills(job_description))

    missing_skills = required_skills - resume_skills

    if not missing_skills:
        return {
            "matched_skills": sorted(required_skills),
            "missing_skills": [],
            "roadmap": [],
            "message": "No missing skills were detected.",
        }

    roadmap = []
    included = set()

    # Add prerequisites first when they are also missing.
    for skill in sorted(missing_skills):
        prerequisites = PREREQUISITES.get(skill, [])

        for prerequisite in prerequisites:
            if (
                prerequisite not in resume_skills
                and prerequisite in missing_skills
                and prerequisite not in included
            ):
                roadmap.append({
                    "skill": prerequisite,
                    "priority": "High",
                    "reason": (
                        f"Learn this foundation before "
                        f"studying {skill}."
                    ),
                    "resource": LEARNING_RESOURCES.get(
                        prerequisite,
                        "Search for a beginner tutorial or official documentation.",
                    ),
                })
                included.add(prerequisite)

        if skill not in included:
            roadmap.append({
                "skill": skill,
                "priority": (
                    "High" if skill in {
                        "python",
                        "sql",
                        "machine learning",
                        "statistics",
                    } else "Medium"
                ),
                "reason": (
                    "This skill appears in the job description "
                    "but was not detected in the resume."
                ),
                "resource": LEARNING_RESOURCES.get(
                    skill,
                    "Search for a beginner tutorial or official documentation.",
                ),
            })
            included.add(skill)

    return {
        "matched_skills": sorted(resume_skills & required_skills),
        "missing_skills": sorted(missing_skills),
        "roadmap": roadmap,
        "message": (
            "Roadmap generated from the detected skill gaps. "
            "Review the suggestions before using them."
        ),
    }


if __name__ == "__main__":

    sample_resume = """
    Python developer familiar with NumPy and pandas.
    """

    sample_job = """
    Required skills: Python, machine learning,
    deep learning, PyTorch, and statistics.
    """

    result = generate_learning_roadmap(
        sample_resume,
        sample_job,
    )

    print("\nMatched skills:")
    print(result["matched_skills"])

    print("\nMissing skills:")
    print(result["missing_skills"])

    print("\nLearning roadmap:")

    for step in result["roadmap"]:
        print(f"\nSkill: {step['skill']}")
        print(f"Priority: {step['priority']}")
        print(f"Reason: {step['reason']}")
        print(f"Resource: {step['resource']}")