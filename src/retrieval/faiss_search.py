import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class InternshipSearch:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.internships = []

    def build_index(self, internships):
        """
        Create embeddings for internship descriptions
        and store them in a FAISS index.
        """

        self.internships = internships

        descriptions = [
            internship["description"]
            for internship in internships
        ]

        embeddings = self.model.encode(descriptions)

        embeddings = np.array(embeddings).astype("float32")

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        print(f"FAISS index created with {len(internships)} internships.")

    def search(self, resume_text, top_k=5):
        """
        Find the most relevant internships for a resume.
        """

        resume_embedding = self.model.encode([resume_text])

        resume_embedding = np.array(
            resume_embedding
        ).astype("float32")

        faiss.normalize_L2(resume_embedding)

        scores, indices = self.index.search(
            resume_embedding,
            top_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            internship = self.internships[index].copy()

            internship["similarity_score"] = float(score)

            results.append(internship)

        return results