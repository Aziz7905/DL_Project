# app/agents/utils/retrievers.py
from __future__ import annotations
import os, pickle
from dataclasses import dataclass
from typing import List, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


try:
    from langchain_huggingface import HuggingFaceEmbeddings  # preferred
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback

from app.config.settings import Config


@dataclass
class _WeightedHit:
    doc: Document
    score: float


class HybridRetriever:
    """
    Minimal hybrid retriever that queries both:
      - dense: FAISS (BGE local embeddings)
      - sparse: BM25 (prebuilt)
    and merges results with weights.
    """
    def __init__(self, faiss_store: FAISS, bm25, k: int = 8, w_bm25: float = 0.5, w_faiss: float = 0.5):
        self.faiss_store = faiss_store
        self.bm25 = bm25
        self.k = int(max(1, k))
        self.w_bm25 = float(w_bm25)
        self.w_faiss = float(w_faiss)

    def invoke(self, query: str) -> List[Document]:
        if not query or not query.strip():
            return []

        # Vector search (FAISS returns docs; treat higher rank = lower score)
        vec_docs: Sequence[Document] = self.faiss_store.similarity_search(query, k=self.k)

        # BM25 search (bm25 is a LangChain retriever/pickle)
        self.bm25.k = self.k
        bm_docs: Sequence[Document] = self.bm25.invoke(query)

        # Score by position (simple, fast). Top-1 gets 1.0, top-k gets ~0.
        def rank_score(i: int, k: int) -> float:
            return 1.0 - (i / max(1, k - 1)) if k > 1 else 1.0

        scored: dict = {}
        for i, d in enumerate(vec_docs):
            s = self.w_faiss * rank_score(i, len(vec_docs))
            key = (d.page_content, tuple(sorted(d.metadata.items())))
            prev = scored.get(key)
            if prev:
                prev.score += s
            else:
                scored[key] = _WeightedHit(doc=d, score=s)

        for i, d in enumerate(bm_docs):
            s = self.w_bm25 * rank_score(i, len(bm_docs))
            key = (d.page_content, tuple(sorted(d.metadata.items())))
            prev = scored.get(key)
            if prev:
                prev.score += s
            else:
                scored[key] = _WeightedHit(doc=d, score=s)

        # Return top-k by combined score
        hits = sorted(scored.values(), key=lambda x: x.score, reverse=True)
        return [h.doc for h in hits[: self.k]]


def load_ensemble_retriever() -> HybridRetriever:
    """
    Loads FAISS and BM25 from disk and returns a HybridRetriever.
    Uses LOCAL HuggingFaceEmbeddings (no network).
    """
    # ---- LOCAL embeddings (no HTTP) ----
    model_name = getattr(Config, "HF_MODEL_EMBED", "BAAI/bge-large-en-v1.5")
    use_cuda = os.environ.get("USE_CUDA", "0") == "1"
    device = "cuda" if use_cuda else "cpu"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        # Optionally add a cache folder if you want to pin local model location:
        # cache_folder=str(Path(".cache/hf_models")),
    )

    if not os.path.exists(Config.INDEX_PATH):
        raise FileNotFoundError(f"❌ FAISS index not found at {Config.INDEX_PATH}. Run the index builder first.")

    faiss_store = FAISS.load_local(
        folder_path=Config.INDEX_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    if not os.path.exists(Config.BM25_PATH):
        raise FileNotFoundError(f"❌ BM25 index not found at {Config.BM25_PATH}. Run the index builder first.")

    with open(Config.BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
        bm25.k = 8

    # You can tune weights later
    return HybridRetriever(faiss_store=faiss_store, bm25=bm25, k=8, w_bm25=0.5, w_faiss=0.5
)
