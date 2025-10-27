"""
Source credibility prior scoring.

- Primary: DeBERTa model from models/deberta (returns score + confidence)
- Fallbacks:
  * Curated table lookup if DeBERTa is not confident
  * Fallback LLM only if domain scoring not found or not confident

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

# --- NEW: Import DeBERTa scorer
from app.models.deberta.scorer import DebertaScorer

# ----------------------------
# Default reputation priors
# ----------------------------
DEFAULT_REPUTATION: Dict[str, float] = {
    # ... (unchanged)
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

MULTI_SUFFIX = (
    # ... (unchanged)
)

def _strip_to_host(s: str) -> str:
    # ... (unchanged)
    pass

def _etld_plus_one(host: str) -> str:
    # ... (unchanged)
    pass

def _normalize_domain(s: Optional[str]) -> str:
    # ... (unchanged)
    pass

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
        # --- NEW: Initialize your DeBERTa scorer
        self.deberta_scorer = DebertaScorer()

    def _lookup_table(self, domain: str) -> Optional[float]:
        # ... (unchanged)
        pass

    def _fallback_score(self, domain: str) -> float:
        # ... (unchanged)
        pass

    def _score_with_deberta(self, domain: str) -> (float, bool):
        """
        Try scoring with DeBERTa.
        Returns: (score, is_confident)
        """
        try:
            score, confidence = self.deberta_scorer.score(domain)
            is_confident = (confidence > 0.7) # Adjust threshold as needed
            return score, is_confident
        except Exception:
            return NEUTRAL_PRIOR, False

    def score_source(self, source_type: Optional[str], source_name: Optional[str]) -> float:
        """
        Returns a credibility prior 1.0–5.0 for a publisher/domain.
        First, try DeBERTa. If not confident, use table, then fallback to LLM.
        """
        if not source_name:
            return NEUTRAL_PRIOR

        domain = _normalize_domain(source_name)
        if not domain:
            return NEUTRAL_PRIOR

        # 1) Try DeBERTa model
        deberta_score, deberta_confident = self._score_with_deberta(domain)
        if deberta_confident:
            return float(f"{max(1.0, min(5.0, deberta_score)):.1f}")

        # 2) Table lookup
        hit = self._lookup_table(domain)
        if hit is not None:
            return float(f"{max(1.0, min(5.0, hit)):.1f}")

        # 3) LLM fallback (optional)
        val = self._fallback_score(domain)
        return float(f"{max(1.0, min(5.0, val)):.1f}")
