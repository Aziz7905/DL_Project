"""
Source credibility prior scoring.

- Persona (fallback LLM): "Expert Media Reliability Researcher"
- Behavior:
  * Normalize domains (strip scheme, path, params; collapse www/subdomains).
  * Look up in a curated table; if missing and fallback is enabled, ask the LLM for a 1.0–5.0 prior.
  * Clamp + round results; return neutral 2.5 on any ambiguity/error.

Rationale:
Short news claims often come from headlines/snippets. A conservative but useful
prior over the *publisher* helps calibrate verification when evidence is thin.
"""

from __future__ import annotations
from typing import Dict, Optional
import re

from langchain_core.messages import HumanMessage
from app.agents.utils.llm import make_mistral_endpoint
from app.config.settings import Config


# ----------------------------
# Default reputation priors
# ----------------------------
DEFAULT_REPUTATION: Dict[str, float] = {
    # Wire & mainstream
    "reuters.com": 4.6,
    "apnews.com": 4.5,
    "bbc.com": 4.3,
    "nytimes.com": 4.2,
    "theguardian.com": 4.1,
    "bloomberg.com": 4.3,
    "wsj.com": 4.2,
    "ft.com": 4.2,

    # Tech/business
    "theverge.com": 3.9,
    "techcrunch.com": 3.8,
    "arstechnica.com": 4.0,
    "engadget.com": 3.7,

    # Company newsrooms (may be reliable for *their own* announcements; still PR)
    "apple.com": 3.6,
    "googleblog.com": 3.6,
    "about.fb.com": 3.4,
    "news.microsoft.com": 3.6,

    # Social & noisy
    "twitter.com": 2.0,
    "tiktok.com": 1.5,
    "reddit.com": 2.3,
}

NEUTRAL_PRIOR = 2.5


# ----------------------------
# LLM fallback prompt (persona-led)
# ----------------------------
PROMPT_REPUTATION_FALLBACK = """
You are an **Expert Media Reliability Researcher**. Your task is to assign a conservative
credibility *prior* (1.0–5.0, one decimal place) to a news **publisher domain**.

### What to consider
- Editorial standards, track record for factual accuracy, corrections policy, reputation among journalists.
- Independence vs. PR/marketing voice (official company newsroom is credible for *its own* announcements but not independent).
- Avoid political bias judgments; focus on verifiability and consistency.
- Heavily user-generated or short-form social video sites score lower as primary sources.

### Rules
- Output **only** the number (e.g., `4.2`) — no words, no units, no explanation.
- Use **one decimal place**.
- Clamp between **1.0** and **5.0**.
- If the domain is unknown, choose a careful neutral-ish value between **2.3** and **2.9**.

[DOMAIN]
{domain}

[NUMBER ONLY]
""".strip()


# ----------------------------
# Domain normalization
# ----------------------------
_SCHEME_RE = re.compile(r"^[a-z]+://", re.I)
_PORT_RE = re.compile(r":\d+$")
_TRAIL_PATH_RE = re.compile(r"[/?#].*$")

# Common multi-label public suffixes we’ll treat as a single TLD unit (simple heuristic)
MULTI_SUFFIX = (
    "co.uk", "gov.uk", "ac.uk",
    "com.au", "net.au", "org.au",
    "co.jp",
    "com.br", "com.mx", "com.tr",
)

def _strip_to_host(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = _SCHEME_RE.sub("", s)            # remove http://, https://
    s = _TRAIL_PATH_RE.sub("", s)        # remove path/query/fragment
    s = s.strip("/")
    s = _PORT_RE.sub("", s)              # remove :port
    if s.startswith("www."):
        s = s[4:]
    return s

def _etld_plus_one(host: str) -> str:
    """
    Best-effort eTLD+1 without external deps.
    """
    if not host:
        return ""
    # If it matches any known multi-suffix, keep 3 labels; else keep last 2 labels
    for suf in MULTI_SUFFIX:
        if host.endswith("." + suf) or host == suf:
            parts = host.split(".")
            if len(parts) >= 3:
                return ".".join(parts[-3:])
            return host
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host

def _normalize_domain(s: Optional[str]) -> str:
    if not s:
        return ""
    # If a filename like "article.pdf, p.3" was passed, strip commas/trailing junk
    s = s.split(",")[0].strip()
    host = _strip_to_host(s)
    return _etld_plus_one(host)


# ----------------------------
# Agent
# ----------------------------
class SourceScorerAgent:
    def __init__(self, table: Optional[Dict[str, float]] = None, enable_fallback_llm: bool = False):
        self.table = (table or DEFAULT_REPUTATION).copy()
        self.enable_fallback_llm = enable_fallback_llm
        self.fallback_llm = make_mistral_endpoint(
            Config.HF_MODEL_EXTRACT, max_new_tokens=8, temperature=0.1
        ) if enable_fallback_llm else None

    def _lookup_table(self, domain: str) -> Optional[float]:
        if not domain:
            return None
        # Exact match
        if domain in self.table:
            return float(self.table[domain])
        # Suffix match (e.g., sub.example.com → example.com)
        etld1 = _etld_plus_one(domain)
        if etld1 in self.table:
            return float(self.table[etld1])
        # Nothing found
        return None

    def _fallback_score(self, domain: str) -> float:
        if not self.enable_fallback_llm or not self.fallback_llm:
            return NEUTRAL_PRIOR
        try:
            msg = self.fallback_llm.invoke([HumanMessage(content=PROMPT_REPUTATION_FALLBACK.format(domain=domain))])
            raw = (msg.content or "").strip()
            # Parse number with one decimal
            try:
                val = float(raw)
            except Exception:
                return NEUTRAL_PRIOR
            # Clamp to [1.0, 5.0]
            val = max(1.0, min(5.0, val))
            # Round to one decimal
            return float(f"{val:.1f}")
        except Exception:
            return NEUTRAL_PRIOR

    def score_source(self, source_type: Optional[str], source_name: Optional[str]) -> float:
        """
        Returns a credibility prior 1.0–5.0 for a publisher/domain.
        - `source_type` is unused here but kept for interface compatibility.
        - `source_name` can be a URL, a filename with a URL-ish prefix, or a bare domain.
        """
        if not source_name:
            return NEUTRAL_PRIOR

        domain = _normalize_domain(source_name)
        if not domain:
            return NEUTRAL_PRIOR

        # 1) Table lookup
        hit = self._lookup_table(domain)
        if hit is not None:
            return float(f"{max(1.0, min(5.0, hit)):.1f}")

        # 2) LLM fallback (optional)
        val = self._fallback_score(domain)
        return float(f"{max(1.0, min(5.0, val)):.1f}")
