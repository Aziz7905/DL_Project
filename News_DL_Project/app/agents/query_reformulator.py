"""
Query reformulation for retrieval / web verification.
- Persona: "Expert Information Retrieval Specialist for Short News"
- Output: Strict JSON with keyword queries, a semantic paraphrase, and preferred domains.
- Optimized for HuffPost-style headline + short description and claim verification.
"""

from __future__ import annotations
from typing import Dict, List
import json
import re
import logging

from langchain_core.messages import HumanMessage

from app.config.settings import Config
from app.agents.utils.llm import make_mistral_endpoint

logger = logging.getLogger(__name__)


# ----------------------------
# Prompt (persona-led, detailed) - FIXED with escaped braces
# ----------------------------
PROMPT_REFORMULATE = """
You are an **Expert Information Retrieval Specialist for Short News**. Your task is to generate
focused search queries that maximize the chance of retrieving *precise, corroborating evidence*
for a user QUESTION (often about a short news item: headline + brief description).

### Operating principles
1) **Intent fidelity**: Preserve the core intent and entities of the QUESTION. Avoid topic drift.
2) **High-precision keyword queries**:
   - Produce **3–5 AND-friendly keyword queries** (no full sentences).
   - Include **canonical entity names**, key verbs, and distinguishing attributes (model, event, region).
   - Add **date/temporal cues** if present (month, year, "next month", "Q4" → normalize to YYYY or YYYY-MM).
   - Prefer terms that disambiguate (e.g., "Vision Pro" + "availability" + "Europe").
   - Avoid stop-words, fluff, punctuation. Use quotes for exact names / models if helpful.
3) **Semantic paraphrase**:
   - Include 1 **semantic_query** (natural-language) that is concise, faithful, and neutral.
   - This query should be *readable* to a human and appropriate for neural retrieval or LLM reranking.
4) **Preferred domains**:
   - Suggest reputable outlets relevant to **tech/business/news verification**.
   - Default set (tune to intent): ["reuters.com", "apnews.com", "bbc.com", "theverge.com", "bloomberg.com"].
   - If the QUESTION includes a brand/product (e.g., Apple), include its newsroom if appropriate
     (e.g., "apple.com/newsroom") **without assuming the answer is there**.
5) **Language**: Match the QUESTION language (EN/FR). Keep keyword queries language-consistent.
6) **No hallucinations**: Do not invent numbers, dates, models, or internal codenames not in QUESTION.
7) **Brevity & control**:
   - Each keyword query ≤ 9 tokens; prefer 5–8 tokens.
   - 3–5 keyword queries total; deduplicate near-duplicates.

### Output (STRICT JSON)
Return **only** valid JSON with this schema:
{{
  "keyword_queries": ["...", "..."],
  "semantic_query": "...",
  "preferred_domains": ["...", "..."]
}}

QUESTION:
{question}

JSON ONLY:
""".strip()


# ----------------------------
# Small utilities
# ----------------------------
def _safe_json_loads(s: str) -> Dict[str, object]:
    """Parse JSON; if extra text leaked, try to extract the first {{...}} object."""
    s = s.strip()
    
    # Remove markdown code blocks if present
    if s.startswith("```"):
        s = re.sub(r'^```(?:json)?\s*', '', s)
        s = re.sub(r'\s*```$', '', s)
        s = s.strip()
    
    # Try direct parse
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    
    # Extract first top-level {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
    
    return {}

_WS_RE = re.compile(r"\s+")

def _norm(q: str) -> str:
    """Normalize whitespace and punctuation."""
    q = str(q).strip()
    q = _WS_RE.sub(" ", q)
    # Remove trailing punctuation commonly emitted
    q = q.strip(" ,;:.")
    return q

def _dedupe_keep_order(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    out: List[str] = []
    for x in items:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

def _limit_tokens(q: str, max_tokens: int = 9) -> str:
    """Limit query to max tokens."""
    toks = q.split()
    if len(toks) <= max_tokens:
        return q
    return " ".join(toks[:max_tokens])


# ----------------------------
# Agent
# ----------------------------
class QueryReformulatorAgent:
    def __init__(self, model_id: str | None = None):
        """
        Uses HF Inference API (conversational) via ChatHuggingFace.
        Low temperature for deterministic structure; slightly higher max tokens for JSON.
        """
        self.llm = make_mistral_endpoint(
            model_id or Config.HF_MODEL_EXTRACT,
            max_new_tokens=400,
            temperature=0.2
        )

    def reformulate(self, question: str) -> Dict[str, List[str] | str]:
        """
        Reformulate a question into optimized search queries.
        
        Returns:
            Dict with keys: keyword_queries, semantic_query, preferred_domains
        """
        # Clean and validate input
        question = _norm(question)
        
        # Fallback defaults
        default_result = {
            "keyword_queries": [question] if question else [],
            "semantic_query": question,
            "preferred_domains": ["reuters.com", "apnews.com", "bbc.com", "theverge.com", "bloomberg.com"]
        }
        
        if not question:
            logger.warning("Empty question provided to reformulate()")
            return default_result
        
        try:
            prompt = PROMPT_REFORMULATE.format(question=question)
            msg = self.llm.invoke([HumanMessage(content=prompt)])
            raw = msg.content.strip()
            
            logger.info(f"Reformulating '{question[:50]}...' -> Raw response: {raw[:200]}...")
            
            data = _safe_json_loads(raw)
            
            if not data:
                logger.warning(f"Failed to parse JSON from response for question: {question}")
                return default_result

            # Initialize fields with safe defaults
            keyword_queries: List[str] = []
            semantic_query: str = question
            preferred_domains: List[str] = default_result["preferred_domains"]

            # Extract + sanitize keyword queries
            if isinstance(data.get("keyword_queries"), list):
                keyword_queries = [
                    _norm(str(x)) 
                    for x in data["keyword_queries"] 
                    if isinstance(x, str) and _norm(str(x))
                ]
            
            # Extract semantic query
            if isinstance(data.get("semantic_query"), str) and _norm(data["semantic_query"]):
                semantic_query = _norm(data["semantic_query"])
            
            # Extract preferred domains
            if isinstance(data.get("preferred_domains"), list):
                tmp = [
                    _norm(str(x)).lower() 
                    for x in data["preferred_domains"] 
                    if isinstance(x, str) and _norm(str(x))
                ]
                if tmp:
                    preferred_domains = tmp

            # Hard constraints: length, count, dedupe
            keyword_queries = [_limit_tokens(q, 9) for q in keyword_queries]
            keyword_queries = [q for q in keyword_queries if 3 <= len(q.split()) <= 9]
            keyword_queries = _dedupe_keep_order(keyword_queries)
            
            if len(keyword_queries) > 5:
                keyword_queries = keyword_queries[:5]
            
            if len(keyword_queries) < 3:
                # Backfill minimal variants from the question if needed (defensive)
                base = _norm(question)
                if base and base not in keyword_queries:
                    keyword_queries.append(_limit_tokens(base, 9))
                # Simple heuristic expansions
                if " vs " in base.lower():
                    keyword_queries.append(_limit_tokens(base.replace(" vs ", " versus "), 9))
                if " and " in base.lower():
                    keyword_queries.append(_limit_tokens(base.replace(" and ", " "), 9))
                keyword_queries = _dedupe_keep_order(keyword_queries)[:3]

            # Domains: ensure sane bounds and dedupe
            preferred_domains = [d for d in preferred_domains if "." in d]
            preferred_domains = _dedupe_keep_order(preferred_domains)[:8]
            if not preferred_domains:
                preferred_domains = ["reuters.com", "apnews.com", "bbc.com"]

            result = {
                "keyword_queries": keyword_queries,
                "semantic_query": semantic_query,
                "preferred_domains": preferred_domains,
            }
            
            logger.info(f"Reformulation successful: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in reformulate(): {e}", exc_info=True)
            return default_result
