# app/models/schemas.py
from typing import Optional
from pydantic import BaseModel, Field


class AnalyzePayload(BaseModel):
    question: Optional[str] = Field(default=None, description="User question; optional if article provided")
    article: Optional[str] = Field(default=None, description="Raw article text to extract/verify claims from")
    session_id: Optional[str] = Field(default=None, description="Short-term/long-term memory session")

    # Behavior knobs
    use_reformulation: bool = True
    do_claims: bool = True
    verify_source_score: bool = True
    use_web: bool = False
    do_explain_scores: bool = False
    k_retrieval: int = 6
    k_ltm: int = 3
    max_claims: int = 5


class FeedbackPayload(BaseModel):
    session_id: Optional[str] = None
    question: Optional[str] = None
    answer_html: Optional[str] = None
    reward: Optional[float] = None  # [-1..+1] or [0..1]
    verdict: Optional[str] = None   # e.g., correct, unclear, misleading
    meta: Optional[dict] = None
