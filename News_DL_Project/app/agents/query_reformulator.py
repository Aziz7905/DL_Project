"""
Query reformulation - FIXED VERSION (escaped JSON in prompt)
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
# Prompt (JSON curly braces ESCAPED with double braces)
# ----------------------------
PROMPT_REFORMULATE = """
You are an expert at reformulating search queries. Given a user question, create better search queries.

Return ONLY valid JSON with this exact structure:
{{
  "keyword_queries": ["query1", "query2", "query3"],
  "semantic_query": "natural language version",
  "preferred_domains": ["reuters.com", "apnews.com", "bbc.com"]
}}

Rules:
- keyword_queries: 3-5 short search queries (5-8 words each)
- semantic_query: A clear, natural rephrasing of the question
- preferred_domains: 3-5 reliable news sources

QUESTION: {question}

JSON OUTPUT:
""".strip()


# ----------------------------
# Utilities
# ----------------------------
def _safe_json_loads(s: str) -> Dict[str, object]:
    """Parse JSON; extract first {{...}} if needed."""
    s = s.strip()
    
    # Try direct parse first
    try:
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)
    except json.JSONDecodeError:
        pass
    
    # Extract JSON block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON in markdown code blocks
    json_block = re.search(r'```(?:json)?\s*(\{{.*?\}})\s*```', s, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except json.JSONDecodeError:
            pass
    
    logger.warning(f"Could not extract JSON from: {s[:200]}")
    return {}

_WS_RE = re.compile(r"\s+")

def _norm(q: str) -> str:
    """Normalize whitespace and punctuation."""
    q = q.strip()
    q = _WS_RE.sub(" ", q)
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
    return " ".join(toks[:max_tokens]) if len(toks) > max_tokens else q


# ----------------------------
# Agent
# ----------------------------
class QueryReformulatorAgent:
    def __init__(self, model_id: str | None = None):
        """Initialize with HF Inference API model."""
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
        # Fallback defaults
        default_result = {
            "keyword_queries": [_norm(question)],
            "semantic_query": _norm(question),
            "preferred_domains": ["reuters.com", "apnews.com", "bbc.com", "theverge.com", "bloomberg.com"]
        }
        
        # Guard against empty input
        if not question or not question.strip():
            logger.warning("Empty question provided to reformulate()")
            return default_result
        
        try:
            # Call LLM
            prompt = PROMPT_REFORMULATE.format(question=question)
            msg = self.llm.invoke([HumanMessage(content=prompt)])
            raw = msg.content.strip()
            
            logger.info(f"Raw LLM response: {raw[:300]}")
            
            # Parse JSON
            data = _safe_json_loads(raw)
            
            if not data:
                logger.warning(f"Failed to parse JSON from response for question: {question}")
                return default_result
            
            # Extract and validate fields
            keyword_queries: List[str] = []
            semantic_query: str = question
            preferred_domains: List[str] = default_result["preferred_domains"]
            
            # Keyword queries
            if isinstance(data.get("keyword_queries"), list):
                keyword_queries = [
                    _norm(str(x)) 
                    for x in data["keyword_queries"] 
                    if isinstance(x, str) and _norm(str(x))
                ]
                keyword_queries = [_limit_tokens(q, 9) for q in keyword_queries]
                keyword_queries = [q for q in keyword_queries if 3 <= len(q.split()) <= 9]
                keyword_queries = _dedupe_keep_order(keyword_queries)[:5]
            
            # Fallback if no valid keyword queries
            if not keyword_queries:
                logger.warning("No valid keyword queries extracted, using original question")
                keyword_queries = [_limit_tokens(_norm(question), 9)]
            
            # Semantic query
            if isinstance(data.get("semantic_query"), str) and _norm(data["semantic_query"]):
                semantic_query = _norm(data["semantic_query"])
            else:
                logger.warning("No semantic query in response, using original")
                semantic_query = _norm(question)
            
            # Preferred domains
            if isinstance(data.get("preferred_domains"), list):
                tmp = [
                    _norm(str(x)).lower() 
                    for x in data["preferred_domains"] 
                    if isinstance(x, str) and _norm(str(x)) and "." in str(x)
                ]
                if tmp:
                    preferred_domains = _dedupe_keep_order(tmp)[:8]
            
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