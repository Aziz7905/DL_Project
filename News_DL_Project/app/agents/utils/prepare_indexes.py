# app/agents/utils/prepare_indexes.py
"""
Index builder (LOCAL embeddings):
- Loads .md/.txt/.pdf/.doc/.docx recursively under Config.DATA_DIR
- Embeddings: LOCAL HuggingFace (BGE-large by default) -> FAISS
- Also builds BM25 retriever
- Chunking: Config.CHUNK_SIZE / Config.CHUNK_OVERLAP

Run:
  python -m app.agents.utils.prepare_indexes
"""

import os
import pickle
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# Prefer the new package; fall back to community if needed.
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # modern
except Exception:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback

from app.config.settings import Config

SUPPORTED_EXTS = {".pdf", ".txt", ".docx", ".doc", ".md"}


def load_documents(data_path: str) -> List[Document]:
    """Recursively load all supported documents under data_path."""
    docs: List[Document] = []
    base = Path(data_path)

    for root, _, files in os.walk(data_path):
        root_path = Path(root)
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext not in SUPPORTED_EXTS:
                continue

            full_path = str(root_path / fname)
            rel_source = str((root_path / fname).relative_to(base))  # for nicer citations

            # Choose loader
            if ext == ".pdf":
                loader = PyPDFLoader(full_path)
            elif ext == ".txt":
                loader = TextLoader(full_path, encoding="utf-8")
            elif ext in (".docx", ".doc"):
                loader = UnstructuredWordDocumentLoader(full_path)
            else:
                loader = UnstructuredMarkdownLoader(full_path)

            try:
                file_docs = loader.load()
                for d in file_docs:
                    d.metadata.setdefault("source", rel_source)
                docs.extend(file_docs)
            except Exception as e:
                print(f"⚠️ Error loading {rel_source}: {e}")

    return docs


def _build_local_embeddings():
    """
    Create a LOCAL embeddings model (no external requests).
    Default to BGE-large unless overridden by Config.HF_MODEL_EMBED.
    """
    model_name = getattr(Config, "HF_MODEL_EMBED", None) or "BAAI/bge-large-en-v1.5"

    # Choose device. For GPU: set environment variable USE_CUDA=1 before running.
    use_cuda = os.environ.get("USE_CUDA", "0") == "1"
    device = "cuda" if use_cuda else "cpu"

    # normalize_embeddings=True is recommended for BGE family
    # trust_remote_code is safe for popular sentence-transformers, but optional.
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    print(f"🔧 Using LOCAL embeddings: '{model_name}' on device '{device}'")
    return embeddings


def process_and_store(
    data_dir: str = Config.DATA_DIR,
    faiss_path: str = Config.INDEX_PATH,
    bm25_path: str = Config.BM25_PATH,
):
    # ---- Local embeddings (no HTTP calls) ----
    embeddings = _build_local_embeddings()

    faiss_store = None
    existing_sources = set()

    if Path(faiss_path).exists():
        print("🔁 Loading existing FAISS index…")
        faiss_store = FAISS.load_local(
            folder_path=faiss_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        # Collect already indexed filenames to avoid re-adding
        try:
            for d in faiss_store.similarity_search(" ", k=2000):
                s = d.metadata.get("source")
                if s:
                    existing_sources.add(s)
        except Exception as e:
            print(f"ℹ️ Could not pre-scan existing index for sources: {e}")

    # Load all docs (recursive)
    if not Path(data_dir).exists():
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    docs = load_documents(data_dir)

    # Only new sources
    new_docs = [d for d in docs if d.metadata.get("source") not in existing_sources]
    if not new_docs:
        print("✅ No new documents to add.")
        return

    print(f"📄 {len(new_docs)} new documents. Chunking…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(new_docs)

    # Deduplicate exact chunk text
    seen = set()
    uniq: List[Document] = []
    for d in chunks:
        c = d.page_content.strip()
        if c and c not in seen:
            seen.add(c)
            uniq.append(d)

    print(f"📦 {len(uniq)} unique chunks ready for indexing.")

    # Build / update FAISS
    if faiss_store is None:
        print("🧱 Building new FAISS index…")
        faiss_store = FAISS.from_documents(uniq, embeddings)
    else:
        print("➕ Adding to existing FAISS index…")
        faiss_store.add_documents(uniq)

    faiss_store.save_local(faiss_path)
    print(f"✅ FAISS index saved at '{faiss_path}/'")

    # Build BM25 over the same chunks (simple & fast)
    print("🧮 Building BM25 retriever…")
    bm25 = BM25Retriever.from_documents(uniq)
    bm25.k = 8
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"✅ BM25 retriever saved at '{bm25_path}'")


if __name__ == "__main__":
    process_and_store()
