# app/agents/qa_chain.py
"""
Grounded(ish) QA chain optimized for short news items (e.g., HuffPost dataset).
- Persona: Expert News Analyst & Information Synthesizer
- Behavior: Prefer details explicitly in CONTEXT, but allow cautious, common-sense inference when context is thin.
- Safety: Do not invent numbers/dates/quotes/entities that aren’t present; note missing pieces succinctly.
"""

import os
from typing import List, Tuple, Optional
from markdown import markdown
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from app.agents.utils.llm import GroqChat
from app.agents.utils.retrievers import load_ensemble_retriever
from app.config.settings import Config
from app.memory.langchain_memory import memory

# ---------------------------------------------------------------------------
# PROMPT: Expert persona + detailed guidance for short news (headline/desc)
# ---------------------------------------------------------------------------

PROMPT_QA = """
You are an **Expert News Analyst & Information Synthesizer** specializing in short news items
(headlines and brief descriptions). You excel at:
- extracting concrete facts and key takeaways,
- giving compact, precise answers,
- paraphrasing safely when the text is short,
- clearly signaling any missing details.

### Your operating principles
1) **Use the CONTEXT as primary evidence.** Prefer exact details found there.
2) **Careful inference is allowed** for obvious implications typical in news headlines/briefs
   (e.g., a launch event implies product announcements), **but do not fabricate** numbers, dates,
   locations, names, quotes, or statistics that are not given.
3) **If something is missing**, say so in a short clause (e.g., “the date isn’t specified”).
4) **Be concise (2–5 sentences)** or a **short bullet list** when appropriate.
5) **Match the QUESTION’s language** (French/English).
6) **No hallucinations.** If a specific detail is not present, do not invent it.
7) **No external knowledge** unless trivially common-sense (e.g., “a launch event is for announcements”).
8) **Neutral tone.** No hype, no opinionated framing.

### Answer format
Choose the single best structure:
- **Short Answer**: 1–3 sentences if the question is direct.
- **Bulleted Summary**: 3–6 bullets (≤ 15 words each) if listing expected items or key points.
- **Mini Synthesis**: 2–5 sentences if the question asks for context or implications.

### When context is thin
- You **may paraphrase** and **infer the obvious** (e.g., “expected to”, “likely about”),
  but keep modality clear (do not turn expectations into facts).
- If the question requests details that are not present (e.g., price, exact date, model numbers),
  state succinctly that they are not specified in the provided text.

================ CONTEXT =================
{context}
==========================================

QUESTION:
{question}

### Your task
Provide the best possible answer using the rules above. Keep it compact and precise.
If any key piece is missing from the CONTEXT, mention that briefly.

=== BEGIN RESPONSE ===
"""

def build_prompt() -> PromptTemplate:
    return PromptTemplate(template=PROMPT_QA, input_variables=["context", "question"])

# Keep singletons so other modules can import them.
_retriever = load_ensemble_retriever()

# Slightly higher temperature than ultra-strict QA to allow safe paraphrase.
_llm = GroqChat(model_name=Config.GROQ_MODEL_ANSWER, temperature=0.3, max_tokens=1024)

_prompt = build_prompt()

def _build_context(query: str, history: str = "", k: int = 6) -> Tuple[str, List[Document]]:
    """
    Build retrieval context:
    - `history` includes STM/LTM (if any), placed before retrieved snippets.
    - For short news, a small k (2–6) is usually enough.
    """
    docs: List[Document] = _retriever.invoke(query)
    texts = [d.page_content for d in docs[:k]]
    ctx = (history.strip() + ("\n\n====\n\n" if history else "") + "\n\n---\n\n".join(texts)).strip()
    return ctx, docs

def _token_overlap(a: str, b: str) -> int:
    """Tiny relevance gate to avoid listing junk sources."""
    stop = {"the","a","an","and","to","of","in","on","for","by","with","at","is","are","it","that","this","as","from"}
    ak = {w for w in a.lower().split() if w not in stop}
    bk = {w for w in b.lower().split() if w not in stop}
    return len(ak & bk)

def answer_question_with_sources(
    query: str,
    conversation=None,
    session_id: Optional[str] = None,
    *,
    article_text: Optional[str] = None,
    k_retrieval: int = 6,
):
    """
    Main entry:
    - If `article_text` is provided → use it as the sole context (article-first grounding).
    - Else builds memory+retrieval context.
    - Applies the News Analyst prompt.
    - Returns formatted HTML + deduplicated, relevance-filtered source labels (if any).
    """
    # 1) Short-term + Long-term memory
    stm_text = memory.stm_to_text(session_id)
    ltm_docs = memory.ltm_recall(session_id, query, k=3)
    ltm_texts = [f"[LTM:{d.metadata.get('kind','note')}] {d.page_content}" for d in ltm_docs]
    ltm_block = "\n".join(ltm_texts).strip()
    history = "\n\n".join([b for b in [stm_text, ltm_block] if b]).strip()

    # 2) Build context
    docs: List[Document] = []
    if article_text and article_text.strip():
        # Article-first: do NOT mix unrelated retrieval into context.
        ctx = (history + ("\n\n====\n\n" if history else "") + article_text.strip()[:8000]).strip()
    else:
        ctx, docs = _build_context(query, history, k=k_retrieval)

    # 3) LLM answer
    filled = _prompt.format(context=ctx, question=query)
    result = _llm.invoke([HumanMessage(content=filled)]).content.strip()

    # 4) Memory writes
    if session_id:
        memory.stm_add_turn(session_id, user_text=query, assistant_text=result)
        answer_only = result.split("\n\n📚 **Sources**")[0]
        memory.ltm_add(session_id, text=query, kind="user_q")
        memory.ltm_add(session_id, text=answer_only[:500], kind="assistant_a")

    # 5) Sources (only when using retrieval; filter out irrelevant junk)
    sources = []
    if docs:
        labels = []
        for d in docs:
            meta = d.metadata or {}
            src = meta.get("source") or meta.get("file_path") or "unknown"
            label = os.path.basename(str(src))
            page = meta.get("page")
            if page is not None:
                label += f", p.{page}"
            # tiny overlap gate to avoid nonsense sources
            if _token_overlap(query, (d.page_content or "")[:500]) >= 2:
                labels.append(label)
        sources = sorted(set(labels))
        if sources:
            result += "\n\n📚 **Sources** : " + ", ".join(sources)

    html = markdown(result, extensions=["fenced_code", "tables", "nl2br"])
    return html, sources
