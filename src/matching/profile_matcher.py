from src.matching.profile_extractor import (
    extract_years_of_experience,
    extract_education
)


def calculate_experience_score(resume_text, job_text):
    """
    Compare candidate experience with internship requirements.
    """

    resume_years = extract_years_of_experience(resume_text)
    required_years = extract_years_of_experience(job_text)

    # If the internship does not specify experience,
    # assume it is suitable for the candidate.
    if required_years == 0:
        return 1.0

    # Candidate meets or exceeds requirement
    if resume_years >= required_years:
        return 1.0

    # Candidate has some experience but less than required
    if resume_years > 0:
        return resume_years / required_years

    # No experience
    return 0.0


def calculate_education_score(resume_text, job_text):
    """
    Compare candidate education with job requirements.
    """

    resume_education = set(
        extract_education(resume_text)
    )

    job_education = set(
        extract_education(job_text)
    )

    # JD doesn't specify education requirements
    if not job_education:
        return 1.0

    # Find common education qualifications
    matched = resume_education.intersection(
        job_education
    )

    if matched:
        return 1.0

    # Candidate has a relevant technical degree
    technical_degrees = {
        "b.tech",
        "b.e",
        "m.tech",
        "m.e",
        "computer science",
        "artificial intelligence",
        "machine learning"
    }

    if resume_education.intersection(technical_degrees):
        return 0.7

    return 0.0


if __name__ == "__main__":

    resume = """
    B.Tech in Computer Science Engineering
    specializing in Artificial Intelligence and Machine Learning.
    """

    internship = """
    Looking for candidates with a B.Tech degree
    in Computer Science or Artificial Intelligence.
    """

    experience_score = calculate_experience_score(
        resume,
        internship
    )

    education_score = calculate_education_score(
        resume,
        internship
    )

    print("Experience Score:", experience_score)
    print("Education Score:", education_score)