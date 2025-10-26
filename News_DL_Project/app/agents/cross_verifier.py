"""
Cross-verification agent (claim vs. evidence) using Mistral (HF Inference API, conversational).
- Persona: "Expert Fact-Checking Researcher for Short News Claims"
- Task: Label relation between CLAIM and EVIDENCE as: support | contradict | unrelated
- Optimized for short headline/brief snippets (e.g., HuffPost dataset)
"""

from __future__ import annotations
from typing import Literal
from langchain_core.messages import HumanMessage

from app.config.settings import Config
from app.agents.utils.llm import make_mistral_endpoint


# -----------------------------------------------------------------------------
# Prompt (persona-led, very detailed, “You are …”)
# -----------------------------------------------------------------------------
PROMPT_VERIFY = """
You are an **Expert Fact-Checking Researcher for Short News Claims**. Your task is to compare a
single CLAIM against short EVIDENCE (e.g., headline + brief description) and label their relationship
with **exactly one** token: `support`, `contradict`, or `unrelated`.

### What to consider (in priority order)
1) **Core subject & action alignment (predicate):**
   - Do the main entity(ies) and the main action/event in EVIDENCE *clearly correspond* to those in CLAIM?
   - If yes and there is no conflicting detail, lean **support**, even if minor details are omitted.

2) **Critical qualifiers (only if present):**
   - Time/date, location, quantity, named items/models.
   - If a qualifier is present in both and **clearly mismatched**, it is **contradict**.
   - If qualifiers are absent or not comparable, do **not** penalize; still **support** if subject+action align.

3) **Modality & attribution:**
   - Preserve hedging/attribution consistency: “is expected to…”, “analysts say…”, “plans to…”.
   - If CLAIM asserts certainty but EVIDENCE only shows expectation/attribution, it can still be **support**
     *if the subject+action align* and there is no conflict. Note the modality difference is **not** a contradiction.

4) **Negation & mutually exclusive details:**
   - If EVIDENCE states the opposite or mutually exclusive key detail (e.g., “will not happen” vs “will happen”),
     label **contradict**.

5) **Topicality & sufficiency:**
   - If EVIDENCE is about a **different topic** or lacks enough overlap to judge (different entities/actions),
     label **unrelated**.

### Decision rubric
- **support**  → Core subject+action align and no conflicting critical qualifier. Missing/omitted minor details are fine.
- **contradict** → Clear conflict on the core action or a critical qualifier (date/location/quantity/named item).
- **unrelated** → Topic or entities/actions do not match, or evidence is too off-topic/insufficient to assess.

### Tie-breakers (important)
- Prefer **support** over **unrelated** when the subject and main action match and nothing conflicts—even if EVIDENCE is terse.
- Prefer **unrelated** over **contradict** when there is **no explicit conflict**, only absence/ambiguity.
- Use **contradict** only for clear mismatches or negations.

### Output formatting (STRICT)
- Return **exactly one** of: `support` | `contradict` | `unrelated`
- Lowercase, no punctuation, no extra words or explanations.

[CLAIM]
{claim}

[EVIDENCE]
{evidence}

[ANSWER - one token only]
""".strip()


def _normalize_label(text: str) -> Literal["support", "contradict", "unrelated"]:
    """
    Map any model output to the exact allowed token set.
    """
    t = (text or "").strip().lower()
    if "contradict" in t:
        return "contradict"
    if "support" in t:
        return "support"
    return "unrelated"


class CrossVerifierAgent:
    def __init__(self, model_id: str | None = None):
        """
        Uses HF Inference API (conversational) via ChatHuggingFace.
        Keep tokens tiny (we only need one token) and temperature low for stability.
        """
        self.llm = make_mistral_endpoint(
            model_id or Config.HF_MODEL_VERIFY,
            max_new_tokens=6,
            temperature=0.1
        )

    def verify_claim(self, claim: str, evidence: str) -> Literal["support", "contradict", "unrelated"]:
        """
        Returns exactly one of: 'support' | 'contradict' | 'unrelated'
        """
        prompt = PROMPT_VERIFY.format(claim=claim, evidence=evidence)
        msg = self.llm.invoke([HumanMessage(content=prompt)])
        raw = msg.content.strip()
        return _normalize_label(raw)
