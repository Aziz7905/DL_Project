# app/agents/orchestrator.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, time
from markdown import markdown
from langchain_core.messages import HumanMessage

from app.agents.qa_chain import _retriever as QA_RETRIEVER, _llm as QA_LLM, _prompt as QA_PROMPT
from app.agents.qa_chain import _build_context as qa_build_context
from app.agents.claim_extractor import ClaimExtractorAgent
from app.agents.cross_verifier import CrossVerifierAgent
from app.agents.evidence_retriever import EvidenceRetrieverAgent
from app.agents.query_reformulator import QueryReformulatorAgent
from app.agents.source_scorer import SourceScorerAgent
from app.agents.aggregator import AggregatorAgent
from app.agents.web_retriever import WebRetrieverAgent  # optional
from app.memory.langchain_memory import memory

class OrchestratorAgent:
    def __init__(self):
        self.retriever = QA_RETRIEVER
        self.qa_llm = QA_LLM
        self.qa_prompt = QA_PROMPT

        self._claim_extractor = None
        self._cross_verifier = None
        self._query_reformulator = None
        self._evidence_agent = EvidenceRetrieverAgent()
        self._source_scorer = SourceScorerAgent()
        self._aggregator = AggregatorAgent()
        self._web_agent = None

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

    def _run_qa(
        self,
        question: str,
        session_id: Optional[str],
        k_retrieval: int,
        k_ltm: int,
        article_text: Optional[str] = None,   # ← added: prime with article text
    ):
        t0 = time.time()

        # 0) Prime with article headline/description (HuffPost-style)
        prime = (article_text or "").strip()
        prime_block = f"[HEADLINE+DESCRIPTION]\n{prime}\n" if prime else ""

        # 1) memory
        stm_text = memory.stm_to_text(session_id)
        ltm_docs = memory.ltm_recall(session_id, question, k=k_ltm)
        ltm_texts = [f"[LTM:{d.metadata.get('kind','note')}] {d.page_content}" for d in ltm_docs]
        ltm_block = "\n".join(ltm_texts).strip()
        history = "\n\n".join([b for b in [stm_text, ltm_block] if b]).strip()

        # 2) retrieval & context
        ctx, docs = qa_build_context(question, history, k=k_retrieval)
        # Prepend the prime block if present
        ctx = "\n\n====\n\n".join([b for b in [prime_block, ctx] if b]).strip()

        filled = self.qa_prompt.format(context=ctx, question=question)

        # 3) ask the LLM (news-friendly QA)
        result = self.qa_llm.invoke([HumanMessage(content=filled)]).content.strip()

        # 4) sources
        sources = []
        if docs:
            labels = []
            for d in docs:
                meta = d.metadata or {}
                src = meta.get("source") or meta.get("file_path") or "unknown"
                page = meta.get("page")
                label = os.path.basename(str(src))
                if page is not None:
                    label += f", p.{page}"
                labels.append(label)
            sources = sorted(set(labels))
            if sources:
                result += "\n\n📚 **Sources** : " + ", ".join(sources)

        html = markdown(result, extensions=["fenced_code", "tables", "nl2br"])

        # 5) memory write
        if session_id:
            memory.stm_add_turn(session_id, user_text=question, assistant_text=result)
            answer_only = result.split("\n\n📚 **Sources**")[0]
            memory.ltm_add(session_id, text=question, kind="user_q")
            memory.ltm_add(session_id, text=answer_only[:500], kind="assistant_a")

        t1 = time.time()
        return {"html": html, "sources": sources, "latency_s": round(t1 - t0, 3)}

    def _extract_claims(self, text: str, max_claims: int) -> List[str]:
        return self._get_claim_extractor().extract_claims(text, max_claims=max_claims)

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

        # Optional web evidence
        web_snips, web_links = [], []
        if use_web:
            try:
                war = self._get_web_agent().get_live_evidence(claim)[:3]
                for r in war:
                    title = (r.get("title") or "").strip()
                    snippet = (r.get("snippet") or "").strip()
                    link = r.get("link") or r.get("url")
                    # Prefer snippet; fall back to title if snippet missing
                    if snippet or title:
                        web_snips.append(snippet if snippet else title)
                        if link:
                            web_links.append(link)
            except Exception:
                pass

        # Compose evidence for verifier (local preferred)
        evidence_text = "\n\n---\n\n".join(local_snips + web_snips)[:4000] or "."

        # Cross verification label
        verdict = self._get_cross_verifier().verify_claim(claim, evidence_text)

        # Source scoring prior
        src_score = None
        if verify_source_score:
            src_score = 2.5
            for s in local_sources:
                score = self._source_scorer.score_source(None, s)
                if score > src_score:
                    src_score = score

        # Heuristic support score mapped to 1..5 scale (softer neutral)
        support_score = 4.3 if verdict == "support" else 1.7 if verdict == "contradict" else 3.0

        final_score = None
        explanation = None
        if verify_source_score:
            final_score = self._aggregator.aggregate(support_score, src_score or 2.5, verdict)
            if do_explain_scores:
                explanation = self._aggregator.explain(
                    claim=claim,
                    support_score=support_score,
                    source_score=src_score or 2.5,
                    cross_verification=verdict,
                    final_score=final_score,
                )

        return {
            "claim": claim,
            "verdict": verdict,  # support | contradict | unrelated
            "support_score": support_score,
            "source_score": src_score,
            "final_score": final_score,
            "explanation": explanation,
            "evidence": {
                "local_snippets": local_snips,
                "local_sources": list(sorted(set(local_sources))),
                "web_snippets": web_snips if use_web else [],
                "web_links": web_links if use_web else [],
            },
        }

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

        timings = {}
        t_all = time.time()

        # Derive working question
        working_question = (question or "").strip()
        if not working_question and article:
            working_question = article.strip().split("\n")[0][:200]

        # Optional query reformulation
        plan = None
        if use_reformulation and working_question:
            t0 = time.time()
            try:
                plan = self._get_reformulator().reformulate(working_question)
            except Exception:
                plan = None
            timings["reformulation_s"] = round(time.time() - t0, 3)

        # QA
        answer = None
        if working_question:
            t0 = time.time()
            answer = self._run_qa(
                working_question,
                session_id,
                k_retrieval=k_retrieval,
                k_ltm=k_ltm,
                article_text=article,  # ← pass article to prime QA
            )
            timings["qa_s"] = round(time.time() - t0, 3)

        # Claims + Verification
        claims: List[str] = []
        verification: List[Dict[str, Any]] = []
        if do_claims:
            source_text = (article or "")
            if not source_text and answer:
                # strip HTML to text (before sources)
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(answer["html"], "html.parser")
                    text = soup.get_text("\n")
                    source_text = text.split("📚 **Sources**")[0]
                except Exception:
                    source_text = ""

            if source_text.strip():
                t0 = time.time()
                claims = self._extract_claims(source_text, max_claims=max_claims)
                timings["claims_s"] = round(time.time() - t0, 3)

                t1 = time.time()
                for c in claims:
                    verification.append(
                        self._verify_claim(
                            claim=c,
                            use_web=use_web,
                            verify_source_score=verify_source_score,
                            do_explain_scores=do_explain_scores,
                        )
                    )
                timings["verification_s"] = round(time.time() - t1, 3)

        timings["total_s"] = round(time.time() - t_all, 3)

        return {
            "answer": answer,
            "claims": claims,
            "verification": verification,
            "plan": plan,
            "meta": {
                "timings": timings,
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
            },
        }
