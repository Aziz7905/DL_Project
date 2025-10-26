# app/memory/langchain_memory.py
"""
Dual memory:
- STM: Last N turns stored in memory per session.
- LTM: FAISS vector store with LOCAL HuggingFace embeddings (created lazily).
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Prefer modern import, fallback if unavailable.
try:
    from langchain_huggingface import HuggingFaceEmbeddings  # modern local embeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback

from app.config.settings import Config


MEMORY_DIR = Path(Config.INDEX_PATH).parent / "memory_store"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
LTM_PATH = str(MEMORY_DIR / "faiss_ltm")


# ---------------- STM (Short-Term Memory) ---------------- #

@dataclass
class _STMEntry:
    """Short-term memory entry holding conversation history."""
    history: ChatMessageHistory
    window_size: int = 4

    def get_recent_messages(self) -> str:
        """Return formatted text of last N message pairs (user+AI)."""
        messages = self.history.messages
        recent = messages[-(self.window_size * 2):] if len(messages) > self.window_size * 2 else messages

        lines: List[str] = []
        for msg in recent:
            if msg.type == "human":
                lines.append(f"Human: {msg.content}")
            elif msg.type == "ai":
                lines.append(f"Assistant: {msg.content}")
        return "\n".join(lines)


# ---------------- Dual Memory System ---------------- #

class DualMemory:
    """Simplified dual memory with STM + LTM."""

    def __init__(self, stm_window_pairs: int = 4):
        self._sessions: Dict[str, _STMEntry] = {}
        self._stm_window_pairs = stm_window_pairs

        # Local embeddings setup
        model_name = getattr(Config, "HF_MODEL_EMBED", "BAAI/bge-large-en-v1.5")
        use_cuda = os.environ.get("USE_CUDA", "0") == "1"
        device = "cuda" if use_cuda else "cpu"

        self._embed = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device, "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )

        # Try to load existing FAISS index; otherwise, create lazily on first add
        if Path(LTM_PATH).exists():
            self._ltm = FAISS.load_local(
                folder_path=LTM_PATH,
                embeddings=self._embed,
                allow_dangerous_deserialization=True,
            )
        else:
            self._ltm = None  # lazily create on first ltm_add

    def _sid(self, session_id: Optional[str]) -> str:
        """Ensure session_id is never None."""
        return session_id or "global"

    # ---- STM Control ---- #
    def stm_get(self, session_id: Optional[str]) -> _STMEntry:
        sid = self._sid(session_id)
        if sid not in self._sessions:
            self._sessions[sid] = _STMEntry(history=ChatMessageHistory(), window_size=self._stm_window_pairs)
        return self._sessions[sid]

    def stm_add_turn(self, session_id: Optional[str], user_text: str, assistant_text: str):
        entry = self.stm_get(session_id)
        entry.history.add_user_message(user_text)
        entry.history.add_ai_message(assistant_text)

    def stm_to_text(self, session_id: Optional[str]) -> str:
        return self.stm_get(session_id).get_recent_messages()

    def stm_clear(self, session_id: Optional[str]):
        sid = self._sid(session_id)
        if sid in self._sessions:
            self._sessions[sid].history.clear()

    # ---- LTM Control ---- #
    def _ensure_loaded(self):
        """Load FAISS from disk if needed (used by recall)."""
        if self._ltm is None and Path(LTM_PATH).exists():
            self._ltm = FAISS.load_local(
                folder_path=LTM_PATH,
                embeddings=self._embed,
                allow_dangerous_deserialization=True,
            )

    def ltm_add(self, session_id: Optional[str], text: str, kind: str = "note", extra: Optional[dict] = None):
        """Store long-term text chunks with metadata."""
        if not text or not text.strip():
            return

        meta = {"session_id": self._sid(session_id), "kind": kind}
        if extra:
            meta.update(extra)

        doc = Document(page_content=text.strip(), metadata=meta)

        # Lazily build the FAISS index on first document
        if self._ltm is None:
            self._ltm = FAISS.from_documents([doc], self._embed)
        else:
            self._ltm.add_documents([doc])

        self._ltm.save_local(LTM_PATH)

    def ltm_recall(self, session_id: Optional[str], query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant long-term memory."""
        self._ensure_loaded()
        if self._ltm is None:
            return []

        retriever = self._ltm.as_retriever(
            search_kwargs={"k": int(max(1, k)), "filter": {"session_id": self._sid(session_id)}}
        )
        return retriever.invoke(query or "") or []

    # ---- Combined Context ---- #
    def get_full_context(self, session_id: Optional[str], query: str = "", ltm_k: int = 3) -> str:
        """Merge STM + LTM context into a single chunk."""
        stm = self.stm_to_text(session_id)
        ltm_docs = self.ltm_recall(session_id, query=query, k=ltm_k) if query else []

        parts = []
        if stm:
            parts.append(f"Recent conversation:\n{stm}")
        if ltm_docs:
            ctx = "\n\n".join([f"- {d.page_content}" for d in ltm_docs])
            parts.append(f"Relevant memory:\n{ctx}")

        return "\n\n".join(parts) if parts else ""


# Singleton instance for global use
memory = DualMemory(stm_window_pairs=4)
