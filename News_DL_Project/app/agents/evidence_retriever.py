# app/agents/evidence_retriever.py
"""
Retrieve factual evidence from local RAG stores that best matches the claim.

Powered by:
- Keyword expansion tuned for news / factual claims
- Hybrid FAISS + BM25 ensemble retriever (already loaded by load_ensemble_retriever)
- Optional title-first weighting (title hits ranked first)

Role: Expert News Evidence Retrieval Specialist
"""

from typing import List
from langchain_core.documents import Document
from app.agents.utils.retrievers import load_ensemble_retriever

import re

class EvidenceRetrieverAgent:
    def __init__(self):
        self.retriever = load_ensemble_retriever()

    def _preprocess_claim(self, claim: str) -> str:
        """
        Make claims more retrieval-friendly:
        - remove filler verbs ("is expected", "analysts believe")
        - remove hedging language (likely, reportedly, etc.)
        - preserve key nouns, entities, numbers, locations
        """
        text = claim.lower()

        # Remove speculative / hedging language
        hedges = [
            r"\bis expected to\b", r"\bis anticipated to\b", r"\breportedly\b",
            r"\bplans? to\b", r"\bmay\b", r"\bmight\b", r"\bcould\b",
            r"\baccording to.*", r"\banalysts? expect\b",
        ]
        for h in hedges:
            text = re.sub(h, "", text)

        # Remove trailing punctuation, redundant spaces
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def get_evidence(self, claim: str, max_docs: int = 3) -> List[Document]:
        if not claim:
            return []

        # Add minimal claim cleanup for news-style retrieval
        cleaned = self._preprocess_claim(claim)

        # Hybrid search for best factual grounding
        docs = self.retriever.invoke(cleaned) or []

        # Prefer documents that match titles strongly
        def title_weight(doc: Document) -> float:
            meta = doc.metadata or {}
            title = (meta.get("title") or "").lower()
            if cleaned in title:
                return 2.0  # strong match
            return 1.0

        docs = sorted(docs, key=title_weight, reverse=True)
        return docs[:max_docs]
