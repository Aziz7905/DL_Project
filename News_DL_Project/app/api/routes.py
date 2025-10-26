# app/api/routes.py
from fastapi import APIRouter
from app.agents.orchestrator import OrchestratorAgent
from app.rl.feedback import log_feedback
from app.models.schemas import AnalyzePayload, FeedbackPayload  

router = APIRouter()
_orch = OrchestratorAgent()


@router.post("/analyze")
def analyze(payload: AnalyzePayload):
    return _orch.analyze(
        question=payload.question,
        article=payload.article,
        session_id=payload.session_id,
        use_reformulation=payload.use_reformulation,
        do_claims=payload.do_claims,
        verify_source_score=payload.verify_source_score,
        use_web=payload.use_web,
        do_explain_scores=payload.do_explain_scores,
        k_retrieval=payload.k_retrieval,
        k_ltm=payload.k_ltm,
        max_claims=payload.max_claims,
    )


@router.post("/feedback")
def feedback(payload: FeedbackPayload):
    log_feedback(payload.dict())
    return {"ok": True}
