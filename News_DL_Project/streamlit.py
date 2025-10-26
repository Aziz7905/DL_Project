# app.py
import os
import json
import requests
import streamlit as st
from html import escape
from urllib.parse import urljoin

st.set_page_config(page_title="News QA & Verification", page_icon="📰", layout="wide")

# ---------------------------
# Config
# ---------------------------
API_URL = os.getenv("ANALYZER_API_URL", "http://127.0.0.1:8000")
ANALYZE_ENDPOINT = "/api/analyze"
RL_ENDPOINT = "/api/rl-feedback"

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.title("⚙️ Settings")
session_id = st.sidebar.text_input("Session ID", value="demo_session_001")
use_reformulation = st.sidebar.checkbox("Use reformulation", value=True)
do_claims = st.sidebar.checkbox("Extract claims", value=True)
verify_source_score = st.sidebar.checkbox("Score sources", value=True)
use_web = st.sidebar.checkbox("Use web for verification", value=True)
do_explain_scores = st.sidebar.checkbox("Explain scores", value=True)
k_retrieval = st.sidebar.slider("k_retrieval", 1, 12, 6)
k_ltm = st.sidebar.slider("k_ltm", 0, 6, 3)
max_claims = st.sidebar.slider("max_claims", 1, 10, 4)

st.sidebar.markdown("---")
st.sidebar.caption(f"POST → {API_URL}{ANALYZE_ENDPOINT}")

# ---------------------------
# Main inputs
# ---------------------------
st.title("📰 Simple News Chat (QA + Claims + Verification)")

col1, col2 = st.columns(2)
with col1:
    question = st.text_input(
        "Question",
        value="What new products is Apple expected to announce at the event?"
    )
with col2:
    st.write("")  # spacing

article = st.text_area(
    "Article / Snippet (optional but recommended)",
    height=180,
    value=(
        "Apple announced that it will hold a major launch event next month in Cupertino. "
        "Analysts expect the company to unveil new MacBook models featuring its latest M3 chip architecture. "
        "Apple is also anticipated to expand availability of the Vision Pro mixed-reality headset "
        "to additional countries in Europe and Asia."
    )
)

run_btn = st.button("Analyze 🔎", type="primary")

# Keep last answer for RL feedback
if "last_answer_html" not in st.session_state:
    st.session_state.last_answer_html = ""
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_session_id" not in st.session_state:
    st.session_state.last_session_id = session_id

# ---------------------------
# Call API
# ---------------------------
if run_btn:
    payload = {
        "question": question or None,
        "article": article or None,
        "session_id": session_id or None,
        "use_reformulation": use_reformulation,
        "do_claims": do_claims,
        "verify_source_score": verify_source_score,
        "use_web": use_web,
        "do_explain_scores": do_explain_scores,
        "k_retrieval": int(k_retrieval),
        "k_ltm": int(k_ltm),
        "max_claims": int(max_claims),
    }

    try:
        url = urljoin(API_URL, ANALYZE_ENDPOINT)
        resp = requests.post(url, json=payload, timeout=45)
    except Exception as e:
        st.error(f"Request error: {e}")
        st.stop()

    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}: {resp.text[:800]}")
        st.stop()

    try:
        data = resp.json()
    except Exception:
        st.error("Failed to parse JSON response.")
        st.text(resp.text[:1000])
        st.stop()

    # ---------------------------
    # Render Answer
    # ---------------------------
    st.subheader("Answer")
    ans = data.get("answer") or {}
    ans_html = ans.get("html") or "<p><em>No answer.</em></p>"

    # store for RL feedback
    st.session_state.last_answer_html = ans_html
    st.session_state.last_question = question
    st.session_state.last_session_id = session_id

    try:
        import streamlit.components.v1 as components
        components.html(
            f"<div style='font-family:system-ui,Segoe UI,Arial,sans-serif'>{ans_html}</div>",
            height=300,
            scrolling=True
        )
    except Exception:
        st.markdown(ans_html, unsafe_allow_html=True)

    meta = data.get("meta", {})
    timings = (meta.get("timings") or {})
    with st.expander("Latency & Knobs"):
        st.json({
            "latency_s": ans.get("latency_s"),
            "timings": timings,
            "knobs": meta.get("knobs")
        })

    # ---------------------------
    # Claims
    # ---------------------------
    if do_claims:
        st.subheader("Extracted Claims")
        claims = data.get("claims") or []
        if not claims:
            st.info("No claims extracted.")
        else:
            for i, c in enumerate(claims, 1):
                st.write(f"**{i}.** {c}")

    # ---------------------------
    # Verification
    # ---------------------------
    verifs = data.get("verification") or []
    st.subheader("Verification")
    if not verifs:
        st.info("No verification results.")
    else:
        for i, v in enumerate(verifs, 1):
            verdict = v.get("verdict") or "unrelated"
            ss = v.get("support_score")
            sc = v.get("source_score")
            fs = v.get("final_score")

            badge = {"support": "✅", "contradict": "❌", "unrelated": "⚪"}.get(verdict, "⚪")
            st.markdown(f"**{i}. {badge} {verdict.capitalize()}** — *{escape(v.get('claim',''))}*")

            cols = st.columns(3)
            cols[0].metric("Support", f"{ss if ss is not None else '-'}")
            cols[1].metric("Source prior", f"{sc if sc is not None else '-'}")
            cols[2].metric("Final", f"{fs if fs is not None else '-'}")

            expl = v.get("explanation")
            if expl:
                st.caption(expl)

            ev = v.get("evidence") or {}
            with st.expander("Evidence"):
                ls = ev.get("local_snippets") or []
                if ls:
                    st.markdown("**Local snippets**")
                    for s in ls:
                        st.write(f"- {s[:400]}{'…' if len(s)>400 else ''}")

                ws = ev.get("web_snippets") or []
                wl = ev.get("web_links") or []
                if ws or wl:
                    st.markdown("**Web**")
                    for idx, (s, link) in enumerate(zip(ws, wl)):
                        st.write(f"- {s[:400]}{'…' if len(s)>400 else ''}")
                        if link:
                            st.markdown(f"  ↪️ [{link}]({link})")

            st.divider()

    # ---------------------------
    # Raw JSON (debug)
    # ---------------------------
    with st.expander("Raw JSON response"):
        st.code(json.dumps(data, indent=2)[:100000], language="json")

else:
    st.info("Enter a question and optional article, then click **Analyze**.")

# ===========================
# RL FEEDBACK (Thumbs)
# ===========================
st.markdown("---")
st.subheader("Feedback (RL)")

col_up, col_down = st.columns(2)
with col_up:
    if st.button("👍 Good Answer"):
        if not st.session_state.last_answer_html:
            st.warning("Run an analysis first.")
        else:
            try:
                url = urljoin(API_URL, RL_ENDPOINT)
                payload = {
                    "session_id": st.session_state.last_session_id,
                    "question": st.session_state.last_question,
                    "answer_html": st.session_state.last_answer_html,
                    "reward": 1,
                    "verdict": "good",
                    "meta": {}
                }
                r = requests.post(url, json=payload, timeout=15)
                if r.status_code == 200:
                    st.success("✅ Feedback recorded.")
                else:
                    st.error(f"RL API error {r.status_code}: {r.text[:300]}")
            except Exception as e:
                st.error(f"RL request failed: {e}")

with col_down:
    if st.button("👎 Bad Answer"):
        if not st.session_state.last_answer_html:
            st.warning("Run an analysis first.")
        else:
            try:
                url = urljoin(API_URL, RL_ENDPOINT)
                payload = {
                    "session_id": st.session_state.last_session_id,
                    "question": st.session_state.last_question,
                    "answer_html": st.session_state.last_answer_html,
                    "reward": -1,
                    "verdict": "bad",
                    "meta": {}
                }
                r = requests.post(url, json=payload, timeout=15)
                if r.status_code == 200:
                    st.info("⚠️ Negative feedback recorded.")
                else:
                    st.error(f"RL API error {r.status_code}: {r.text[:300]}")
            except Exception as e:
                st.error(f"RL request failed: {e}")
