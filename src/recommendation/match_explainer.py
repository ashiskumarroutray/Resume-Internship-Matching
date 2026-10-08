def generate_match_explanation(
    title,
    company,
    final_score,
    semantic_score,
    skill_score,
    experience_score,
    education_score,
    matched_skills,
    missing_skills
):
    """
    Generate a structured explanation for an internship match.
    """

    print("\n" + "=" * 70)
    print("MATCH EXPLANATION")
    print("=" * 70)

    print(f"\nInternship: {title}")
    print(f"Company: {company}")

    print(
        f"\nOverall Match: "
        f"{final_score * 100:.2f}%"
    )

    print("\nWhy this internship matches:")

    if semantic_score >= 0.70:
        print("  ✓ Strong semantic similarity between resume and internship.")

    elif semantic_score >= 0.50:
        print("  ✓ Moderate semantic similarity.")

    else:
        print("  ⚠️ Low semantic similarity.")

    if skill_score >= 0.70:
        print("  ✓ Strong skill alignment.")

    elif skill_score >= 0.40:
        print("  ✓ Moderate skill alignment.")

    else:
        print("  ⚠️ Several required skills are missing.")

    if experience_score >= 0.70:
        print("  ✓ Experience profile is compatible.")

    else:
        print("  ⚠️ Experience requirements may be a gap.")

    if education_score >= 0.70:
        print("  ✓ Educational background is relevant.")

    else:
        print("  ⚠️ Educational alignment is weak.")

    print("\nMatched Skills:")

    if matched_skills:
        for skill in matched_skills:
            print(f"  ✓ {skill}")
    else:
        print("  None")

    print("\nSkills to Improve:")

    if missing_skills:
        for skill in missing_skills:
            print(f"  → {skill}")
    else:
        print("  No major skill gaps identified.")

    return {
        "title": title,
        "company": company,
        "match_score": final_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }