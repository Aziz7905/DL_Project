import os
import json
import requests
import streamlit as st
from urllib.parse import urljoin
from html import escape

st.set_page_config(page_title="📰 News Analyzer", layout="wide")

API_URL = os.getenv("ANALYZER_API_URL", "http://127.0.0.1:8000")
ANALYZE_ENDPOINT = "/api/analyze"
RL_ENDPOINT = "/api/feedback"

# Default session
if "session_id" not in st.session_state:
    st.session_state.session_id = "demo"

st.title("📰 News Credibility Assistant")

# ------------------- Inputs -------------------
question = st.text_input("User Question", value="What new products is Apple expected to announce?")
article = st.text_area(
    "Paste a news snippet (optional)",
    height=160,
    value="Apple will hold a launch event next month in Cupertino featuring new M3 devices."
)

colA, colB, colC = st.columns(3)
use_web = colA.checkbox("🌍 Web verification", value=True)
use_reformulation = colB.checkbox("🔁 Reformulate", value=True)
do_explain_scores = colC.checkbox("ℹ Explain scoring", value=False)

analyze_btn = st.button("Analyze 🔎")

# =================== Call API ===================
if analyze_btn:
    payload = {
        "question": question or None,
        "article": article or None,
        "session_id": st.session_state.session_id,
        "use_web": use_web,
        "use_reformulation": use_reformulation,
        "do_explain_scores": do_explain_scores,
        # let backend defaults handle other knobs
    }
    try:
        resp = requests.post(urljoin(API_URL, ANALYZE_ENDPOINT), json=payload, timeout=90)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"❌ API connection failed: {e}")
        st.stop()

    result = resp.json()

    # Store for RL
    st.session_state.last_question = question
    st.session_state.last_answer_html = result.get("answer", {}).get("html", "")

    # ------------------- Answer -------------------
    st.subheader("💬 Answer")
    st.markdown(result.get("answer", {}).get("html", ""), unsafe_allow_html=True)

    # ------------------- Reformulation -------------------
    ref = result.get("meta", {}).get("reformulation", {})
    if ref.get("used"):
        with st.expander("🔁 Reformulation"):
            st.markdown(f"**Before:** `{ref.get('before') or ''}`")
            st.markdown(f"**After:** `{ref.get('after') or ''}`")
            kqs = ref.get("keyword_queries") or []
            doms = ref.get("preferred_domains") or []
            if kqs:
                st.markdown("**Keyword queries:**")
                for q in kqs:
                    st.code(q)
            if doms:
                st.markdown("**Preferred domains:** " + ", ".join(doms))

    # ------------------- Memory -------------------
    mem = result.get("meta", {}).get("memory", {})
    with st.expander("🧠 Memory"):
        st.markdown("**Short-Term Memory**")
        st.text(mem.get("stm") or "—")
        st.markdown("**Long-Term Retrieved Memory**")
        st.text(mem.get("ltm") or "—")

    # ------------------- Claims & Verification -------------------
    claims = result.get("claims", [])
    verifs = result.get("verification", [])
    if claims:
        st.subheader("🧩 Extracted Claims & Verification")
        for i, (cl, chk) in enumerate(zip(claims, verifs), start=1):
            badge = {"support": "✅", "contradict": "❌", "unrelated": "⚪"}.get(chk.get("verdict"), "⚪")
            st.markdown(f"**{i}. {badge} {cl}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Support", chk.get("support_score"))
            c2.metric("Source", chk.get("source_score"))
            c3.metric("Final", chk.get("final_score"))

            if chk.get("explanation"):
                st.caption(chk["explanation"])

            evid = chk.get("evidence", {})
            with st.expander("🧾 Evidence"):
                # Local evidence with filenames
                loc_snips = evid.get("local_snippets", [])
                loc_srcs = evid.get("local_sources", [])
                if loc_snips:
                    st.markdown("**📍 Local Evidence**")
                    for sn, src in zip(loc_snips, loc_srcs):
                        st.write(f"- *{src}* — {escape(sn[:250])}…")

                # Web evidence with links
                web_snips = evid.get("web_snippets", [])
                web_links = evid.get("web_links", [])
                if web_snips:
                    st.markdown("**🌍 Web Evidence**")
                    for sn, link in zip(web_snips, web_links):
                        st.write(f"- {escape(sn[:220])}…")
                        if link:
                            st.markdown(f"  ↪️ [{link}]({link})")

    # ------------------- Timings & Raw -------------------
    with st.expander("⏱ Latency & Knobs"):
        st.json(result.get("meta", {}))

    with st.expander("🛠 Raw Response JSON"):
        st.json(result)

# =================== RL Feedback ===================
st.markdown("---")
st.subheader("Reward this response")

cols = st.columns(2)
with cols[0]:
    if st.button("👍 Helpful"):
        if not st.session_state.last_answer_html:
            st.warning("Run analysis first.")
        else:
            try:
                requests.post(urljoin(API_URL, RL_ENDPOINT), json={
                    "session_id": st.session_state.session_id,
                    "question": st.session_state.last_question,
                    "answer_html": st.session_state.last_answer_html,
                    "reward": 1,
                    "verdict": "good"
                }, timeout=15)
                st.success("✅ Thanks! Reward logged.")
            except Exception as e:
                st.error(f"RL error: {e}")

with cols[1]:
    if st.button("👎 Not Helpful"):
        if not st.session_state.last_answer_html:
            st.warning("Run analysis first.")
        else:
            try:
                requests.post(urljoin(API_URL, RL_ENDPOINT), json={
                    "session_id": st.session_state.session_id,
                    "question": st.session_state.last_question,
                    "answer_html": st.session_state.last_answer_html,
                    "reward": -1,
                    "verdict": "bad"
                }, timeout=15)
                st.error("⚠️ Feedback recorded.")
            except Exception as e:
                st.error(f"RL error: {e}")
