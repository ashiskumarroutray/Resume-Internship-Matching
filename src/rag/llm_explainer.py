import os

from openai import OpenAI


def generate_ai_explanation(match_context):
    """
    Generate a grounded AI explanation from the structured
    internship match report.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set."
        )

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are an AI career intelligence assistant.

Analyze the following internship match report.

IMPORTANT RULES:
1. Use ONLY the information provided in the report.
2. Do not invent skills, experience, education, or requirements.
3. Explain why the candidate matches the internship.
4. Clearly identify the candidate's skill gaps.
5. Give practical recommendations for improving the match.
6. Keep the explanation concise and professional.

{match_context}

Return the response using exactly these sections:

MATCH SUMMARY
WHY YOU MATCH
SKILL GAPS
RECOMMENDATION
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    sample_context = """
INTERNSHIP MATCH REPORT

Internship:
NLP Intern

Company:
LanguageLabs

Overall Match Score:
82.00%

SCORING BREAKDOWN:
- Semantic Similarity: 88.00%
- Skill Compatibility: 75.00%
- Experience Compatibility: 100.00%
- Education Compatibility: 100.00%

MATCHED SKILLS:
python, nlp, tensorflow

MISSING SKILLS:
docker, aws
"""

    explanation = generate_ai_explanation(
        sample_context
    )

    print("\n" + "=" * 70)
    print("AI MATCH EXPLANATION")
    print("=" * 70)
    print(explanation)