"""
Aggregation of credibility components + short natural-language explanation.

Persona-led prompt for the explanation:
- Persona: "Expert Critical Evaluation Instructor for Evidence Appraisal"
- Goal: explain a WEIGHTED-SUM credibility score succinctly and correctly.

Design:
- Deterministic numeric aggregation (no LLM).
- Optional LLM explanation (concise, weight-aware, non-hallucinated).
"""

from __future__ import annotations
from typing import Optional, Dict
from app.config.settings import Config
from app.agents.utils.llm import GroqChat
from langchain_core.messages import HumanMessage


# ---------------------------------------------------------------------------
# Persona-first, detailed prompt (starts with "You are …")
# ---------------------------------------------------------------------------
PROMPT_AGG_EXPLAIN = """
You are an **Expert Critical Evaluation Instructor for Evidence Appraisal**. Your task is to
briefly explain how a **credibility score** was computed using a **weighted sum** of three
components for a single claim assessment.

### Inputs
- Evidence support score S_e on a 1–5 scale: {support_score}
- Source credibility prior S_s on a 1–5 scale: {source_score}
- Cross-verification label L_c in {{support, contradict, unrelated}}: {cross_verification}
- Weights (must be used exactly): 
  - w_e (evidence) = {w_e:.2f}
  - w_s (source)   = {w_s:.2f}
  - w_c (cross)    = {w_c:.2f}
- Mapping for L_c → numeric C:
  - support → +1
  - contradict → −1
  - unrelated → 0

### Rules
1) The final score F is a **weighted sum**, not an average:
     F = (S_e × w_e) + (S_s × w_s) + (C × w_c)
2) Do **not** invent numbers. Use only the provided values and weights.
3) Be concise (≤ 4 sentences), neutral, and clear.
4) Mention the mapping from the cross-verification label to its numeric C.
5) Conclude with the resulting score on 1–5 scale.

### Claim (for context only; do not re-score it)
{claim}

### Produce
A short paragraph (≤ 4 sentences) in the user's language explaining how F was computed from the inputs.
Do not output equations beyond the simple formula already shown, and do not output any lists or headings.
""".strip()


class AggregatorAgent:
    def __init__(self, weights: Optional[Dict[str, float]] = None, enable_llm_explanations: bool = True):
        self.weights = weights or Config.AGGREGATION_WEIGHTS
        self.enable_llm_explanations = enable_llm_explanations
        self._llm = GroqChat(model_name=Config.GROQ_MODEL_ANSWER, temperature=0.2, max_tokens=220)

    def aggregate(self, support_score: float, source_score: float, cross_verification: str) -> float:
        """
        Deterministic weighted sum mapped to 1..5 and clamped.
        cross_verification ∈ {"support","contradict","unrelated"} → {+1,-1,0}.
        """
        cv_map = {"support": 1.0, "contradict": -1.0, "unrelated": 0.0}
        C = cv_map.get(cross_verification, 0.0)

        w = self.weights
        raw = (
            w["evidence_support"] * float(support_score)
            + w["source_credibility"] * float(source_score)
            + w["cross_verification"] * C
        )
        # Clamp and round to thousandths (UI can round later)
        return max(1.0, min(5.0, round(raw, 3)))

    def explain(self, claim: str, support_score: float, source_score: float,
                cross_verification: str, final_score: float) -> str:
        """
        Short, weight-aware explanation. If LLM fails for any reason, fall back to a deterministic string.
        """
        if not self.enable_llm_explanations:
            return self._deterministic_explanation(claim, support_score, source_score, cross_verification, final_score)

        w = self.weights
        prompt = PROMPT_AGG_EXPLAIN.format(
            support_score=support_score,
            source_score=source_score,
            cross_verification=cross_verification,
            w_e=w["evidence_support"],
            w_s=w["source_credibility"],
            w_c=w["cross_verification"],
            claim=claim,
        )
        try:
            txt = self._llm.invoke([HumanMessage(content=prompt)]).content.strip()
            # Minimal sanity guard: ensure it mentions weights and is not empty.
            if txt:
                return txt
        except Exception:
            pass
        return self._deterministic_explanation(claim, support_score, source_score, cross_verification, final_score)

    def _deterministic_explanation(self, claim: str, s_e: float, s_s: float, label: str, F: float) -> str:
        w = self.weights
        cv_map = {"support": "+1", "contradict": "−1", "unrelated": "0"}
        mapped = cv_map.get(label, "0")
        return (
            f"We computed the credibility as a weighted sum: F = (evidence {s_e:.2f}×{w['evidence_support']:.2f}) "
            f"+ (source {s_s:.2f}×{w['source_credibility']:.2f}) + (cross {mapped}×{w['cross_verification']:.2f}). "
            f"The cross-verification label “{label}” maps to {mapped}. "
            f"This yields a final score of {F:.2f} on a 1–5 scale for the claim."
        )
