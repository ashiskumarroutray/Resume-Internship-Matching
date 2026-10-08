import os
import pandas as pd

from src.preprocessing.resume_dataset_loader import (
    load_resume_dataset,
    get_resume_by_index
)

from src.retrieval.faiss_search import InternshipSearch

from src.matching.keyword_matcher import (
    keyword_match_score
)

from src.matching.hybrid_matcher import (
    calculate_skill_score,
    calculate_final_score
)

from src.matching.profile_matcher import (
    calculate_experience_score,
    calculate_education_score
)


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)


def main():

    # =========================================================
    # LOAD RESUME
    # =========================================================

    print("=" * 70)
    print("LOADING RESUME")
    print("=" * 70)

    resume_df = load_resume_dataset()

    resume_index = 0

    resume = get_resume_by_index(
        resume_df,
        resume_index
    )

    resume_text = resume["text"]

    print(f"\nResume ID: {resume['id']}")
    print(f"Category: {resume['category']}")


    # =========================================================
    # LOAD INTERNSHIPS
    # =========================================================

    csv_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "internships",
        "internships.csv"
    )

    internship_df = pd.read_csv(csv_path)

    internships = internship_df.to_dict(
        "records"
    )

    print(
        f"Internships loaded: "
        f"{len(internships)}"
    )


    # =========================================================
    # BUILD FAISS
    # =========================================================

    search_engine = InternshipSearch()

    search_engine.build_index(
        internships
    )


    # =========================================================
    # SEMANTIC SEARCH
    # =========================================================

    semantic_results = search_engine.search(
        resume_text,
        top_k=len(internships)
    )


    # =========================================================
    # COMPARE ALL INTERNSHIPS
    # =========================================================

    results = []

    for result in semantic_results:

        job_text = result["description"]


        # -----------------------------------------------------
        # TRADITIONAL KEYWORD MATCHING
        # -----------------------------------------------------

        (
            keyword_score,
            matched_keywords,
            missing_keywords
        ) = keyword_match_score(
            resume_text,
            job_text
        )


        # -----------------------------------------------------
        # OUR SKILL MATCHING
        # -----------------------------------------------------

        (
            skill_score,
            matched_skills,
            missing_skills
        ) = calculate_skill_score(
            resume_text,
            job_text
        )


        # -----------------------------------------------------
        # SEMANTIC SCORE
        # -----------------------------------------------------

        semantic_score = (
            result["similarity_score"]
        )


        # -----------------------------------------------------
        # PROFILE SCORES
        # -----------------------------------------------------

        experience_score = (
            calculate_experience_score(
                resume_text,
                job_text
            )
        )

        education_score = (
            calculate_education_score(
                resume_text,
                job_text
            )
        )


        # -----------------------------------------------------
        # HYBRID SCORE
        # -----------------------------------------------------

        hybrid_score = calculate_final_score(
            semantic_score=semantic_score,
            skill_score=skill_score,
            experience_score=experience_score,
            education_score=education_score
        )


        results.append({

            "title": result["title"],

            "company": result["company"],

            "keyword_score": keyword_score,

            "semantic_score": semantic_score,

            "skill_score": skill_score,

            "hybrid_score": hybrid_score,

            "matched_keywords": matched_keywords,

            "matched_skills": matched_skills,

            "missing_skills": missing_skills
        })


    # =========================================================
    # DISPLAY COMPARISON
    # =========================================================

    print("\n")
    print("=" * 90)
    print("KEYWORD vs SEMANTIC vs HYBRID MATCHING")
    print("=" * 90)


    for rank, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 90)

        print(
            f"#{rank} "
            f"{result['title']}"
        )

        print(
            f"Company: "
            f"{result['company']}"
        )

        print(
            f"\nTraditional Keyword Score: "
            f"{result['keyword_score'] * 100:.2f}%"
        )

        print(
            f"Semantic Similarity Score: "
            f"{result['semantic_score'] * 100:.2f}%"
        )

        print(
            f"Skill Matching Score: "
            f"{result['skill_score'] * 100:.2f}%"
        )

        print(
            f"Final Hybrid Score: "
            f"{result['hybrid_score'] * 100:.2f}%"
        )

        print("\nMatched Skills:")

        if result["matched_skills"]:

            for skill in result["matched_skills"]:
                print(f"  ✓ {skill}")

        else:
            print("  None")


        print("\nMissing Skills:")

        if result["missing_skills"]:

            for skill in result["missing_skills"]:
                print(f"  ✗ {skill}")

        else:
            print("  None")


    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print("\n")
    print("=" * 90)
    print("MATCHING APPROACH SUMMARY")
    print("=" * 90)

    print("""
Traditional Keyword Matching:
- Exact word overlap
- Cannot understand context
- Sensitive to wording differences

Semantic Matching:
- Sentence Transformer embeddings
- Understands contextual similarity
- FAISS enables efficient retrieval

Our Hybrid Matching:
- Semantic similarity
- Skill compatibility
- Experience compatibility
- Education compatibility
- Produces explainable final score
""")


if __name__ == "__main__":
    main()