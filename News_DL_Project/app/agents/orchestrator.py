# COMPLETE FIX for orchestrator.py
# This preserves both original and reformulated queries

from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, time
from markdown import markdown
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage

from app.agents.qa_chain import (
    _retriever as QA_RETRIEVER,
    _llm as QA_LLM,
    _prompt as QA_PROMPT,
)
from app.agents.qa_chain import _build_context as qa_build_context
from app.agents.claim_extractor import ClaimExtractorAgent
from app.agents.cross_verifier import CrossVerifierAgent
from app.agents.evidence_retriever import EvidenceRetrieverAgent
from app.agents.query_reformulator import QueryReformulatorAgent
from app.agents.source_scorer import SourceScorerAgent
from app.agents.aggregator import AggregatorAgent
from app.agents.web_retriever import WebRetrieverAgent
from app.memory.langchain_memory import memory


class OrchestratorAgent:
    def __init__(self):
        # wires for QA
        self.retriever = QA_RETRIEVER
        self.qa_llm = QA_LLM
        self.qa_prompt = QA_PROMPT

        # lazy sub-agents
        self._claim_extractor = None
        self._cross_verifier = None
        self._query_reformulator = None
        self._web_agent = None

        # always-available utilities
        self._evidence_agent = EvidenceRetrieverAgent()
        self._source_scorer = SourceScorerAgent()
        self._aggregator = AggregatorAgent()

    # ------------------------ Lazy inits ------------------------
    def _get_claim_extractor(self):
        if self._claim_extractor is None:
            self._claim_extractor = ClaimExtractorAgent()
        return self._claim_extractor

    def _get_cross_verifier(self):
        if self._cross_verifier is None:
            self._cross_verifier = CrossVerifierAgent()
        return self._cross_verifier

    def _get_reformulator(self):
        if self._query_reformulator is None:
            self._query_reformulator = QueryReformulatorAgent()
        return self._query_reformulator

    def _get_web_agent(self):
        if self._web_agent is None:
            self._web_agent = WebRetrieverAgent()
        return self._web_agent

    # =======================================================================
    # QA - FIXED to handle both original and reformulated queries
    # =======================================================================
    def _run_qa(
        self,
        question: str,
        session_id: Optional[str],
        k_retrieval: int,
        k_ltm: int,
        article_text: Optional[str] = None,
        original_question: Optional[str] = None,  # ← NEW parameter
    ) -> Dict[str, Any]:
        t0 = time.time()

        # Use original question for memory operations if provided
        memory_question = original_question or question

        # Memory recall - use ORIGINAL question for better matching
        stm = memory.stm_to_text(session_id)
        ltm_docs = memory.ltm_recall(session_id, memory_question, k=k_ltm)
        ltm_txt = "\n".join([d.page_content for d in ltm_docs]).strip()
        history = "\n\n".join(x for x in [stm, ltm_txt] if x)

        # Retrieval → context (use reformulated question for better retrieval)
        ctx, docs = qa_build_context(question, history, k=k_retrieval)

        # ENSURE history is in context
        if history and "CONVERSATION HISTORY" not in ctx and "conversation" not in ctx.lower()[:200]:
            ctx = f"CONVERSATION HISTORY:\n{history}\n\n{'='*50}\n\nRETRIEVED DOCUMENTS:\n{ctx}"

        # Prompt + LLM
        filled = self.qa_prompt.format(context=ctx, question=question)
        result_text = self.qa_llm.invoke([HumanMessage(content=filled)]).content.strip()

        # HTML
        html = markdown(result_text, extensions=["fenced_code", "tables", "nl2br"])

        # Memory write - use ORIGINAL question
        if session_id:
            memory.stm_add_turn(session_id, memory_question, result_text.split("\n\n📚")[0])
            memory.ltm_add(session_id, memory_question, kind="user_q")
            memory.ltm_add(session_id, result_text[:500], kind="assistant_a")

        return {
            "html": html,
            "raw": result_text,
            "docs": docs,
            "latency_s": round(time.time() - t0, 3),
        }

    # =======================================================================
    # Verification (unchanged)
    # =======================================================================
    def _verify_claim(
        self,
        claim: str,
        use_web: bool,
        verify_source_score: bool,
        do_explain_scores: bool,
    ) -> Dict[str, Any]:

        # Local evidence
        local_docs = self._evidence_agent.get_evidence(claim, max_docs=3)
        local_snips = [d.page_content for d in local_docs]
        local_sources = []
        for d in local_docs:
            meta = d.metadata or {}
            src = meta.get("source") or meta.get("file_path") or "unknown"
            page = meta.get("page")
            name = os.path.basename(str(src))
            if page is not None:
                name += f", p.{page}"
            local_sources.append(name)

        # Web evidence
        web_snips, web_links = [], []
        if use_web:
            try:
                results = self._get_web_agent().get_live_evidence(claim)[:3]
                for r in results:
                    snip = (r.get("snippet") or r.get("title") or "").strip()
                    link = r.get("link") or r.get("url")
                    if snip:
                        web_snips.append(snip)
                    if link:
                        web_links.append(link)
            except Exception:
                pass

        # Compose evidence text for verifier
        evidence_text = "\n\n---\n\n".join(local_snips + web_snips)[:4000] or "."
        verdict = self._get_cross_verifier().verify_claim(claim, evidence_text)

        # Source prior
        src_score = None
        if verify_source_score:
            src_score = max([2.5] + [self._source_scorer.score_source(None, s) for s in local_sources])

        # Heuristic support score
        support_score = 4.3 if verdict == "support" else 1.7 if verdict == "contradict" else 3.0

        final_score = None
        explanation = None
        if verify_source_score:
            final_score = self._aggregator.aggregate(support_score, src_score or 2.5, verdict)
            if do_explain_scores:
                explanation = (
                    f"support={support_score}, source={src_score}, verdict={verdict} → final={final_score}"
                )

        return {
            "claim": claim,
            "verdict": verdict,
            "support_score": support_score,
            "source_score": src_score,
            "final_score": final_score,
            "explanation": explanation,
            "evidence": {
                "local_snippets": local_snips,
                "local_sources": list(dict.fromkeys(local_sources)),
                "web_snippets": web_snips if use_web else [],
                "web_links": web_links if use_web else [],
            },
        }

    # =======================================================================
    # Orchestration - FIXED to preserve original question
    # =======================================================================
    def analyze(
        self,
        question: Optional[str],
        article: Optional[str],
        session_id: Optional[str],
        *,
        use_reformulation: bool = True,
        do_claims: bool = True,
        verify_source_score: bool = True,
        use_web: bool = False,
        do_explain_scores: bool = False,
        k_retrieval: int = 6,
        k_ltm: int = 3,
        max_claims: int = 5,
    ) -> Dict[str, Any]:

        timings: Dict[str, float] = {}
        t_all = time.time()

        # Working question (fallback to article headline)
        working_q = (question or "").strip()
        if not working_q and article:
            working_q = article.strip().split("\n")[0][:200]

        # Keep original for memory operations
        original_q = working_q

        # ---------------- Reformulation ----------------
        plan = None
        reform_after = working_q  # This will be used for retrieval
        if use_reformulation and reform_after:
            t0 = time.time()
            try:
                plan_dict = self._get_reformulator().reformulate(reform_after)
                if isinstance(plan_dict, dict):
                    plan = plan_dict
                    # Prefer the semantic paraphrase for QA/retrieval
                    sa = (plan_dict.get("semantic_query") or "").strip()
                    if sa:
                        reform_after = sa  # Only used for retrieval, NOT memory
            except Exception:
                plan = None
            timings["reformulation_s"] = round(time.time() - t0, 3)

        # ---------------- QA - pass BOTH queries ----------------
        t0 = time.time()
        answer = self._run_qa(
            reform_after or working_q,  # For retrieval
            session_id,
            k_retrieval,
            k_ltm,
            article,
            original_question=original_q  # For memory operations
        )
        timings["qa_s"] = round(time.time() - t0, 3)

        # ---------------- Claims + Verification ----------------
        claims: List[str] = []
        verification: List[Dict[str, Any]] = []
        if do_claims:
            source_text = (article or "").strip()
            if not source_text:
                try:
                    soup = BeautifulSoup(answer["html"], "html.parser")
                    source_text = (soup.get_text("\n") or "").strip()
                except Exception:
                    source_text = ""

            if source_text:
                t0 = time.time()
                claims = self._get_claim_extractor().extract_claims(source_text, max_claims=max_claims)
                timings["claims_s"] = round(time.time() - t0, 3)

                t0 = time.time()
                for c in claims:
                    verification.append(
                        self._verify_claim(
                            claim=c,
                            use_web=use_web,
                            verify_source_score=verify_source_score,
                            do_explain_scores=do_explain_scores,
                        )
                    )
                timings["verification_s"] = round(time.time() - t0, 3)

        timings["total_s"] = round(time.time() - t_all, 3)

        # ---------------- Meta for UI ----------------
        meta: Dict[str, Any] = {
            "memory": {
                "stm": memory.stm_to_text(session_id),
                "ltm": "\n".join(
                    [d.page_content for d in memory.ltm_recall(session_id, original_q, k=k_ltm)]
                ),
            },
            "timings": timings,
            "reformulation": {
                "used": bool(use_reformulation),
                "before": original_q,  # ← Show ORIGINAL question
                "after": reform_after,
                "keyword_queries": (plan or {}).get("keyword_queries") if isinstance(plan, dict) else None,
                "preferred_domains": (plan or {}).get("preferred_domains") if isinstance(plan, dict) else None,
            },
            "web_used": bool(use_web),
            "knobs": {
                "use_reformulation": use_reformulation,
                "do_claims": do_claims,
                "verify_source_score": verify_source_score,
                "use_web": use_web,
                "do_explain_scores": do_explain_scores,
                "k_retrieval": k_retrieval,
                "k_ltm": k_ltm,
                "max_claims": max_claims,
            },
        }

        return {
            "answer": {"html": answer["html"], "latency_s": answer["latency_s"]},
            "claims": claims,
            "verification": verification,
            "plan": plan,
            "meta": meta,
        }
