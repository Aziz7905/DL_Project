"""
Claim extraction with Mistral (HF Inference API, conversational) + rigorous persona prompt.
Optimized for short news items (e.g., HuffPost headline + short_description).

- Persona: "Expert Investigative News Analyst for Factual Claim Extraction"
- Output: Strict JSON array of strings
- Modality: Preserve expectations/hedging if present ("is expected to", "analysts say", etc.)
- Post-processing: deduplicate, trim length, remove artifacts, reject vacuous lines
"""

from __future__ import annotations
from typing import List
import json
import re

from langchain_core.messages import HumanMessage

from app.config.settings import Config
from app.agents.utils.llm import make_mistral_endpoint


# ----------------------------
# Tunable constraints
# ----------------------------
MAX_WORDS_PER_CLAIM = 28
MAX_ARTICLE_LEN = 8000  # chars
MIN_CHARS_CLAIM = 6     # avoid tiny junk like "content="
FORBIDDEN_PREFIXES = (
    "article:", "begin claims", "headline:", "description:", "context:", "metadata:",
    "system role", "[", "{", "<", "claim:", "output:", "json:"
)


# ----------------------------
# Prompt
# ----------------------------
PROMPT_EXTRACT = """
You are an **Expert Investigative News Analyst for Factual Claim Extraction**.
You read a short news item (headline + brief description) and extract a small set of
**concise, verifiable claims** exactly as the text supports. Your job is to identify
what the article actually asserts or attributes (including expectations or forecasts)
without adding or changing details.

### Core principles
1) **Faithfulness to text**: Use only what the article states or clearly attributes.
   Do **not** invent numbers, dates, quotes, names, or locations.
2) **Modality preservation**: If the text expresses uncertainty or attribution
   (e.g., "analysts expect", "is expected to", "plans to"), **keep that modality**.
   Do **not** convert expectations into facts.
3) **Atomicity**: Each claim should be a single, standalone proposition with a clear
   subject and predicate (add key qualifiers like date/place/quantity **only if present**).
4) **Conciseness**: ≤ {max_words} words per claim, one sentence, no trailing commentary.
5) **De-duplication**: Merge or drop near-duplicates; prefer the clearest phrasing.
6) **Scope hygiene**: Ignore boilerplate (author bios, cookie banners, navigation UI),
   unrelated links, and platform artifacts.
7) **No opinions** unless the article itself explicitly attributes the opinion
   (e.g., “experts say …”). If so, preserve the attribution.
8) **No external world knowledge** beyond common-sense syntax/grammar.

### Output requirements (STRICT)
- Return **only** a **JSON array of strings**. Example:
  ["Claim A", "Claim B"]
- No headings, no keys, no preface, no trailing commentary, no markdown.
- 0–{max_claims} items (do not invent to fill slots).

### Examples (style only)
Input: "Analysts expect Apple to unveil M3 MacBooks at next month’s Cupertino event."
Valid claims:
- "Analysts expect Apple to unveil M3 MacBooks at an event next month in Cupertino."

Input: "The bill would ban device repair restrictions, sponsors said."
Valid claims:
- "Sponsors say the bill would ban restrictions on repairing devices."

Now process the ARTICLE below and return the JSON array of claims.

[ARTICLE]
{article}

[JSON OUTPUT ONLY]
""".strip()


# ----------------------------
# Helpers
# ----------------------------
_JSON_LIST_RE = re.compile(r"\[[\s\S]*\]")

def _coerce_json_list(s: str) -> List[str]:
    """
    Try to parse a JSON array from s.
    If the model leaked extra text, extract the first [...] block.
    Fallback: split lines (rarely needed with the strict prompt).
    """
    m = _JSON_LIST_RE.search(s)
    payload = m.group(0) if m else s
    try:
        data = json.loads(payload)
        out: List[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    out.append(item.strip())
                elif isinstance(item, dict):
                    # Accept { "claim": "..."}-style fallbacks if they appear.
                    c = item.get("claim") or item.get("text") or item.get("value")
                    if c:
                        out.append(str(c).strip())
        return [x for x in out if x]
    except Exception:
        # Very defensive fallback
        lines = [ln.strip("-•\t ").strip() for ln in payload.splitlines()]
        return [ln for ln in lines if ln]

def _clean_artifacts(t: str) -> str:
    """
    Remove common repr/prefix artifacts that occasionally leak through chains.
    """
    t = t.replace("content='", "").replace('content="', "")
    t = t.replace("\\n", "\n")
    return t

def _is_vacuous(c: str) -> bool:
    """
    Filter out junky or vacuous lines.
    """
    if not c or len(c) < MIN_CHARS_CLAIM:
        return True
    lc = c.lower().strip()
    if any(lc.startswith(pref) for pref in FORBIDDEN_PREFIXES):
        return True
    # Overly meta or non-claim-y
    if lc in {"n/a", "none", "no claims", "no claim"}:
        return True
    # Looks like pure boilerplate
    if "cookie" in lc and "policy" in lc:
        return True
    return False

def _normalize_claim(c: str) -> str:
    """
    Normalize spacing/punctuation and enforce word budget.
    """
    c = " ".join(c.split())  # collapse whitespace
    # Trim trailing junk punctuation
    c = c.strip(" \t\r\n-•*")
    # Enforce single sentence feel; strip trailing commas/semicolons
    c = c.rstrip(",;: ")
    words = c.split()
    if len(words) > MAX_WORDS_PER_CLAIM:
        c = " ".join(words[:MAX_WORDS_PER_CLAIM]).rstrip(",.;:") + "…"
    # Uppercase first letter if it looks like a sentence
    if c and c[0].islower():
        c = c[0].upper() + c[1:]
    return c

def _dedupe(claims: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for c in claims:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ----------------------------
# Agent
# ----------------------------
class ClaimExtractorAgent:
    def __init__(self, model_id: str | None = None):
        """
        Uses HF Inference API (conversational) via ChatHuggingFace (see make_mistral_endpoint).
        Keep temperature low for determinism; max tokens sized to JSON output.
        """
        self.llm = make_mistral_endpoint(model_id or Config.HF_MODEL_EXTRACT,
                                         max_new_tokens=200,
                                         temperature=0.1)

    def extract_claims(self, article_text: str, max_claims: int = 5) -> List[str]:
        if not article_text or not article_text.strip():
            return []

        capped_article = _clean_artifacts(article_text.strip()[:MAX_ARTICLE_LEN])

        prompt = PROMPT_EXTRACT.format(
            article=capped_article,
            max_words=MAX_WORDS_PER_CLAIM,
            max_claims=max_claims
        )

        msg = self.llm.invoke([HumanMessage(content=prompt)])
        raw = _clean_artifacts(msg.content.strip())

        items = _coerce_json_list(raw)

        # Sanitize, filter, normalize, dedupe
        cleaned: List[str] = []
        for c in items:
            c = _clean_artifacts(c).strip()
            if _is_vacuous(c):
                continue
            c = _normalize_claim(c)
            if _is_vacuous(c):
                continue
            cleaned.append(c)

        cleaned = _dedupe(cleaned)

        # Hard limit count; do not pad
        return cleaned[:max_claims]
