def generate_skill_gap(matched_skills, missing_skills):
    """
    Generate a skill-gap report with priority levels.
    """

    print("\n" + "=" * 60)
    print("SKILL GAP ANALYSIS")
    print("=" * 60)

    print("\nSkills you already have:")

    if matched_skills:
        for skill in matched_skills:
            print(f"  ✓ {skill}")
    else:
        print("  None identified")

    print("\nSkills you are missing:")

    if missing_skills:
        for skill in missing_skills:
            print(f"  ✗ {skill}")
    else:
        print("  No major skill gaps identified")

    # Simple priority system
    high_priority = []
    medium_priority = []

    for skill in missing_skills:
        if skill in [
            "python",
            "java",
            "machine learning",
            "deep learning",
            "nlp",
            "sql",
            "tensorflow",
            "pytorch"
        ]:
            high_priority.append(skill)
        else:
            medium_priority.append(skill)

    print("\nHigh Priority Skills:")
    if high_priority:
        for skill in high_priority:
            print(f"  🔴 {skill}")
    else:
        print("  None")

    print("\nMedium Priority Skills:")
    if medium_priority:
        for skill in medium_priority:
            print(f"  🟡 {skill}")
    else:
        print("  None")

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "high_priority": high_priority,
        "medium_priority": medium_priority
    }