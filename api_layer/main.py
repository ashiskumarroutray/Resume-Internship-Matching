
import os
import tempfile
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.matching.hybrid_matcher import (
    calculate_skill_score,
    calculate_final_score,
)
from src.preprocessing.resume_parser import extract_text_from_pdf
from src.preprocessing.resume_dataset_loader import load_resume_dataset
from src.retrieval.faiss_search import InternshipSearch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERNSHIP_CSV = (
    PROJECT_ROOT / "data" / "raw" / "internships" / "internships.csv"
)

app = FastAPI(
    title="Resume Internship Matching API",
    description="API for resume and internship matching",
    version="1.0.0",
)

# Allows the frontend to call this API from a different local port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_internships():
    if not INTERNSHIP_CSV.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Internship CSV not found: {INTERNSHIP_CSV}",
        )

    df = pd.read_csv(INTERNSHIP_CSV)
    required = {"job_id", "title", "company", "description"}

    if not required.issubset(df.columns):
        raise HTTPException(
            status_code=500,
            detail="Internship CSV must contain job_id, title, company, description.",
        )

    df = df.dropna(subset=["description"])
    return df.to_dict(orient="records")


async def get_input_text(file: UploadFile | None, text: str | None) -> str:
    if text and text.strip():
        return text.strip()

    if file is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF or enter text first.",
        )

    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF uploads are supported. You can paste text instead.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
            temp.write(contents)
            temp_path = temp.name

        extracted = extract_text_from_pdf(temp_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read this PDF. Please check the file or paste its text.",
        ) from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    if not extracted.strip():
        raise HTTPException(
            status_code=400,
            detail="No readable text was found in the PDF.",
        )

    return extracted


def score_candidate(resume_text: str, job_text: str, semantic_score: float) -> dict:
    # Cosine similarity can be below zero; map it into the 0–1 range.
    semantic_score = max(0.0, min(1.0, semantic_score))

    skill_score, matched, missing = calculate_skill_score(
        resume_text, job_text
    )

    final_score = calculate_final_score(
        semantic_score=semantic_score,
        skill_score=skill_score,
        experience_score=0.5,
        education_score=0.5,
    )

    return {
        "score": final_score,
        "semantic": round(semantic_score * 100, 2),
        "coverage": round(skill_score * 100, 2),
        "matched": matched,
        "missing": missing,
        "explanation": (
            f"Matched {len(matched)} required skills; "
            f"{len(missing)} required skills are missing."
        ),
    }


@app.get("/")
def home():
    return {"message": "Resume Internship Matching API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/match-resume")
async def match_resume(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
):
    resume_text = await get_input_text(file, text)
    internships = read_internships()

    if not internships:
        return {"results": []}

    try:
        searcher = InternshipSearch()
        searcher.build_index(internships)
        semantic_results = searcher.search(
            resume_text,
            top_k=min(10, len(internships)),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Internship semantic search failed: {exc}",
        ) from exc

    results = []
    for job in semantic_results:
        details = score_candidate(
            resume_text,
            job["description"],
            job.get("similarity_score", 0.0),
        )
        results.append({
            "title": job.get("title", "Internship"),
            "company": job.get("company", "Unknown company"),
            **details,
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return {"results": results}


@app.post("/api/match-internship")
async def match_internship(
    file: UploadFile | None = File(default=None),
    text: str | None = Form(default=None),
):
    job_text = await get_input_text(file, text)

    try:
        resumes = load_resume_dataset()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load the resume dataset: {exc}",
        ) from exc

    resume_texts = resumes["Resume_str"].tolist()
    if not resume_texts:
        return {"results": []}

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(resume_texts)
        embeddings = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        query = model.encode([job_text])
        query = np.asarray(query, dtype="float32")
        faiss.normalize_L2(query)

        scores, indices = index.search(
            query,
            min(50, len(resume_texts)),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Candidate semantic search failed: {exc}",
        ) from exc

    results = []
    for similarity, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        resume = resumes.iloc[int(idx)]
        candidate_text = str(resume["Resume_str"])
        details = score_candidate(
            candidate_text,
            job_text,
            float(similarity),
        )

        results.append({
            "name": f"Candidate {resume['ID']}",
            "score": details["score"],
            "matched": details["matched"],
            "missing": details["missing"],
            "semantic": details["semantic"],
            "coverage": details["coverage"],
            "category": str(resume["Category"]),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return {"results": results}
