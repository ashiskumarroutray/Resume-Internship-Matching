import re


def clean_text(text):
    """
    Convert text to lowercase and remove unnecessary characters.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#.\- ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def keyword_match_score(resume_text, job_text):
    """
    Calculate traditional keyword overlap score.

    Score =
        matched unique keywords / unique job keywords
    """

    resume_words = set(
        clean_text(resume_text).split()
    )

    job_words = set(
        clean_text(job_text).split()
    )

    if not job_words:
        return 0.0, [], []

    matched_keywords = resume_words.intersection(
        job_words
    )

    missing_keywords = job_words - resume_words

    score = (
        len(matched_keywords)
        / len(job_words)
    )

    return (
        score,
        sorted(matched_keywords),
        sorted(missing_keywords)
    )


if __name__ == "__main__":

    resume = """
    Python developer with experience in machine learning,
    NLP, TensorFlow and SQL.
    """

    internship = """
    Looking for a Python machine learning intern
    with NLP, TensorFlow and SQL experience.
    """

    score, matched, missing = keyword_match_score(
        resume,
        internship
    )

    print("=" * 60)
    print("KEYWORD MATCHING")
    print("=" * 60)

    print(
        f"\nKeyword Match Score: "
        f"{score * 100:.2f}%"
    )

    print("\nMatched Keywords:")

    for word in matched:
        print(f"  ✓ {word}")

    print("\nMissing Keywords:")

    for word in missing:
        print(f"  ✗ {word}")