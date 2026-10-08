import os
import pandas as pd

from src.preprocessing.resume_dataset_loader import (
    load_resume_dataset,
    get_resume_by_index
)

from src.retrieval.faiss_search import InternshipSearch

from src.matching.hybrid_matcher import (
    calculate_skill_score,
    calculate_final_score
)

from src.matching.profile_matcher import (
    calculate_experience_score,
    calculate_education_score
)

from src.recommendation.skill_gap import generate_skill_gap


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)


def main():

    # ============================================================
    # 1. LOAD RESUME DATASET
    # ============================================================

    print("\n" + "=" * 70)
    print("LOADING RESUME DATASET")
    print("=" * 70)

    resume_df = load_resume_dataset()

    # For now, test with the first resume
    resume_index = 0

    resume = get_resume_by_index(
        resume_df,
        index=resume_index
    )

    resume_text = resume["text"]

    print(f"\nResume ID: {resume['id']}")
    print(f"Resume Category: {resume['category']}")
    print(f"Resume characters: {len(resume_text)}")


    # ============================================================
    # 2. LOAD INTERNSHIP DATASET
    # ============================================================

    print("\n" + "=" * 70)
    print("LOADING INTERNSHIP DATASET")
    print("=" * 70)

    csv_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "internships",
        "internships.csv"
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Internship dataset not found at: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required_columns = [
        "job_id",
        "title",
        "company",
        "description"
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Required internship column '{column}' "
                f"not found in internships.csv"
            )

    internships = df.to_dict("records")

    print(f"Internships loaded: {len(internships)}")


    # ============================================================
    # 3. BUILD FAISS INDEX
    # ============================================================

    print("\n" + "=" * 70)
    print("BUILDING FAISS INDEX")
    print("=" * 70)

    search_engine = InternshipSearch()

    search_engine.build_index(
        internships
    )


    # ============================================================
    # 4. SEMANTIC RETRIEVAL
    # ============================================================

    print("\n" + "=" * 70)
    print("SEARCHING FOR BEST INTERNSHIPS")
    print("=" * 70)

    results = search_engine.search(
        resume_text,
        top_k=min(5, len(internships))
    )


    # ============================================================
    # 5. HYBRID MATCHING
    # ============================================================

    ranked_results = []

    for result in results:

        job_text = result["description"]

        # --------------------------------------------------------
        # Skill matching
        # --------------------------------------------------------

        (
            skill_score,
            matched_skills,
            missing_skills
        ) = calculate_skill_score(
            resume_text,
            job_text
        )

        # --------------------------------------------------------
        # Semantic score
        # --------------------------------------------------------

        semantic_score = result[
            "similarity_score"
        ]

        # --------------------------------------------------------
        # Experience score
        # --------------------------------------------------------

        experience_score = calculate_experience_score(
            resume_text,
            job_text
        )

        # --------------------------------------------------------
        # Education score
        # --------------------------------------------------------

        education_score = calculate_education_score(
            resume_text,
            job_text
        )

        # --------------------------------------------------------
        # Final hybrid score
        # --------------------------------------------------------

        final_score = calculate_final_score(
            semantic_score=semantic_score,
            skill_score=skill_score,
            experience_score=experience_score,
            education_score=education_score
        )

        ranked_results.append({

            "job_id": result["job_id"],

            "title": result["title"],

            "company": result["company"],

            "description": result["description"],

            "semantic_score": semantic_score,

            "skill_score": skill_score,

            "experience_score": experience_score,

            "education_score": education_score,

            "final_score": final_score,

            "matched_skills": matched_skills,

            "missing_skills": missing_skills

        })


    # ============================================================
    # 6. SORT RESULTS
    # ============================================================

    ranked_results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )


    # ============================================================
    # 7. DISPLAY RESULTS
    # ============================================================

    print("\n" + "=" * 80)
    print("AI RESUME–INTERNSHIP HYBRID MATCHING")
    print("=" * 80)

    print(f"\nResume ID: {resume['id']}")
    print(f"Resume Category: {resume['category']}")

    print(
        f"\nTop {len(ranked_results)} internship recommendations:"
    )


    for rank, result in enumerate(
        ranked_results,
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"#{rank} - {result['title']}"
        )

        print(
            f"Company: {result['company']}"
        )

        print(
            f"Job ID: {result['job_id']}"
        )

        print(
            f"Semantic Score: "
            f"{result['semantic_score'] * 100:.2f}%"
        )

        print(
            f"Skill Score: "
            f"{result['skill_score'] * 100:.2f}%"
        )

        print(
            f"Experience Score: "
            f"{result['experience_score'] * 100:.2f}%"
        )

        print(
            f"Education Score: "
            f"{result['education_score'] * 100:.2f}%"
        )

        print(
            f"FINAL MATCH SCORE: "
            f"{result['final_score'] * 100:.2f}%"
        )

        print(
            "\nMatched Skills:"
        )

        if result["matched_skills"]:

            for skill in result["matched_skills"]:
                print(f"  ✓ {skill}")

        else:
            print("  None")


        print(
            "\nMissing Skills:"
        )

        if result["missing_skills"]:

            for skill in result["missing_skills"]:
                print(f"  ✗ {skill}")

        else:
            print("  No major skill gaps identified")


        # --------------------------------------------------------
        # Skill Gap Analysis
        # --------------------------------------------------------

        generate_skill_gap(
            result["matched_skills"],
            result["missing_skills"]
        )


    print("\n" + "=" * 80)
    print("MATCHING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()