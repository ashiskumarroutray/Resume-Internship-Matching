import sys

from src.matching.skill_normalizer import normalize_skills
from src.matching.hybrid_matcher import (
    extract_skills,
    calculate_skill_score,
    calculate_final_score,
)
from src.evaluation.ranking_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.recommendation.learning_roadmap import (
    generate_learning_roadmap,
)


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAILED: {name}")

    print(f"PASS: {name}")


def main():
    print("\n=== BACKEND SMOKE TEST ===\n")

    # 1. Skill normalization
    normalized = normalize_skills(["ML", "machine learning", "NLP"])

    check(
        "Skill normalization",
        "machine learning" in normalized
        and "natural language processing" in normalized,
    )

    # 2. Skill extraction
    skills = extract_skills(
        "Python developer familiar with machine learning and PyTorch"
    )

    check(
        "Skill extraction",
        "python" in skills and "pytorch" in skills,
    )

    # 3. Skill coverage
    score, matched, missing = calculate_skill_score(
        "Python and SQL developer",
        "Required: Python, SQL, Docker",
    )

    check(
        "Skill coverage",
        0 <= score <= 1
        and "python" in matched
        and "docker" in missing,
    )

    # 4. Hybrid score
    final_score = calculate_final_score(
        semantic_score=0.8,
        skill_score=0.7,
        experience_score=0.5,
        education_score=0.5,
    )

    check(
        "Hybrid scoring",
        0 <= final_score <= 100,
    )

    # 5. Ranking metrics
    ranking = ["job2", "job1", "job3"]
    relevant = {"job1"}

    check(
        "Precision@2",
        precision_at_k(ranking, relevant, 2) == 0.5,
    )

    check(
        "Recall@2",
        recall_at_k(ranking, relevant, 2) == 1.0,
    )

    check(
        "Reciprocal rank",
        reciprocal_rank(ranking, relevant) == 0.5,
    )

    # 6. Learning roadmap
    roadmap = generate_learning_roadmap(
        "Python developer",
        "Python, machine learning, deep learning, PyTorch",
    )

    check(
        "Learning roadmap",
        isinstance(roadmap, dict)
        and "roadmap" in roadmap
        and "missing_skills" in roadmap,
    )

    print("\nAll smoke tests passed.")
    print(
        "Note: This checks component behavior, not real-world "
        "matching accuracy or full dataset integration."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nBackend smoke test failed: {error}")
        sys.exit(1)