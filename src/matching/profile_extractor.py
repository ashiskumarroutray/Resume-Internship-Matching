import re


def extract_years_of_experience(text):
    """
    Extract approximate years of experience from text.
    """

    text = text.lower()

    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+of\s+experience",
        r"(\d+(?:\.\d+)?)\+?\s*years?\s+experience"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return float(match.group(1))

    return 0.0


def extract_education(text):
    """
    Extract common education qualifications.
    """

    text = text.lower()

    education = []

    education_patterns = {
        "b.tech": [
            "b.tech",
            "btech",
            "bachelor of technology"
        ],

        "b.e": [
            "b.e",
            "be",
            "bachelor of engineering"
        ],

        "m.tech": [
            "m.tech",
            "mtech",
            "master of technology"
        ],

        "m.e": [
            "m.e",
            "master of engineering"
        ],

        "computer science": [
            "computer science",
            "cse"
        ],

        "artificial intelligence": [
            "artificial intelligence",
            "ai"
        ],

        "machine learning": [
            "machine learning",
            "aiml"
        ]
    }

    for education_name, keywords in education_patterns.items():

        for keyword in keywords:

            if keyword in text:
                education.append(education_name)
                break

    return list(set(education))


if __name__ == "__main__":

    sample_text = """
    B.Tech in Computer Science Engineering
    specializing in Artificial Intelligence and Machine Learning.
    """

    print(
        "Years of Experience:",
        extract_years_of_experience(sample_text)
    )

    print(
        "Education:",
        extract_education(sample_text)
    )