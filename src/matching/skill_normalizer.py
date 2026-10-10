import re

SKILL_ALIASES = {
    # Machine Learning
    "ml": "machine learning",
    "machine learning": "machine learning",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "scikit-learn": "scikit-learn",

    # Deep Learning
    "dl": "deep learning",
    "deep learning": "deep learning",

    # NLP
    "nlp": "natural language processing",
    "natural language processing": "natural language processing",

    # Artificial Intelligence
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "generative ai": "generative ai",
    "genai": "generative ai",

    # Frameworks
    "tf": "tensorflow",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",

    # Databases
    "postgres": "postgresql",
    "postgres sql": "postgresql",
    "postgresql": "postgresql",

    # JavaScript ecosystem
    "js": "javascript",
    "javascript": "javascript",
    "node": "node.js",
    "node.js": "node.js",
}


def normalize_skill(skill):
    """Convert a skill or alias to its canonical representation."""

    if not isinstance(skill, str):
        return ""

    skill = re.sub(r"\s+", " ", skill.lower().strip())

    return SKILL_ALIASES.get(skill, skill)


def normalize_skills(skills):
    """Normalize skills and remove duplicates."""

    normalized = {
        normalize_skill(skill)
        for skill in skills
    }

    normalized.discard("")

    return sorted(normalized)


if __name__ == "__main__":

    test_skills = [
        "Python",
        "ML",
        "machine learning",
        "sklearn",
        "NLP",
        "natural language processing",
        "TensorFlow",
        "TF",
        "GenAI",
        "Postgres",
    ]

    print("Original:", test_skills)
    print("Normalized:", normalize_skills(test_skills))