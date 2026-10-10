import pandas as pd
from src.matching.hybrid_matcher import extract_skills

# Load datasets
resumes = pd.read_csv(
    "data/processed/resumes_processed.csv"
).reset_index(drop=True)

jobs = pd.read_csv(
    "data/processed/internships_processed.csv"
)

# Pairs labeled as relevant
relevant_pairs = [
    (0, 1),
    (1, 2),
    (1, 4),
    (2, 3),
]

for resume_id, job_id in relevant_pairs:
    resume = resumes.iloc[resume_id]

    job = jobs.loc[
        jobs["job_id"].astype(str) == str(job_id)
    ].iloc[0]

    resume_text = str(resume["Resume_str"])
    job_text = f"{job['title']} {job['description']}"

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched = set(resume_skills) & set(job_skills)
    missing = set(job_skills) - set(resume_skills)

    print("\n" + "=" * 50)
    print(f"Resume ID: {resume_id}")
    print(f"Internship ID: {job_id}")
    print("Resume skills:", sorted(resume_skills))
    print("Job skills:", sorted(job_skills))
    print("Matched skills:", sorted(matched))
    print("Missing skills:", sorted(missing))