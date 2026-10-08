def build_match_context(
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
    Build structured context for the GenAI explanation layer.
    """

    context = f"""
INTERNSHIP MATCH REPORT

Internship:
{title}

Company:
{company}

Overall Match Score:
{final_score * 100:.2f}%

SCORING BREAKDOWN:
- Semantic Similarity: {semantic_score * 100:.2f}%
- Skill Compatibility: {skill_score * 100:.2f}%
- Experience Compatibility: {experience_score * 100:.2f}%
- Education Compatibility: {education_score * 100:.2f}%

MATCHED SKILLS:
{", ".join(matched_skills) if matched_skills else "None"}

MISSING SKILLS:
{", ".join(missing_skills) if missing_skills else "None"}
"""

    return context.strip()


if __name__ == "__main__":

    context = build_match_context(
        title="NLP Intern",
        company="LanguageLabs",
        final_score=0.82,
        semantic_score=0.88,
        skill_score=0.75,
        experience_score=1.0,
        education_score=1.0,
        matched_skills=[
            "python",
            "nlp",
            "tensorflow"
        ],
        missing_skills=[
            "docker",
            "aws"
        ]
    )

    print(context)