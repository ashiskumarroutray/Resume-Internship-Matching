SKILL_ALIASES = {
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "natural language processing": "nlp",
    "machine learning": "machine learning",
    "ml": "machine learning",
    "deep learning": "deep learning",
    "dl": "deep learning",
    "generative ai": "generative ai",
    "genai": "generative ai",
    "artificial intelligence": "ai",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "pytorch": "pytorch",
    "postgres": "postgresql",
    "postgres sql": "postgresql",
    "js": "javascript",
    "node": "node.js",
}


def normalize_skill(skill):
    """
    Convert a skill/alias into a standard representation.
    """

    skill = skill.lower().strip()

    return SKILL_ALIASES.get(skill, skill)


def normalize_skills(skills):
    """
    Normalize a list of skills.
    """

    return list({
        normalize_skill(skill)
        for skill in skills
    })


if __name__ == "__main__":

    test_skills = [
        "Python",
        "ML",
        "sklearn",
        "NLP",
        "TensorFlow"
    ]

    print("Original:", test_skills)
    print("Normalized:", normalize_skills(test_skills))