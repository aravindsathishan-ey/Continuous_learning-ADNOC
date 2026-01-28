import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime
from rag import FaissStore, FeedbackRAG, Settings
# import rag as ac



SETTINGS = Settings()
VECTORDB = FaissStore(SETTINGS)
RAG = FeedbackRAG(SETTINGS, VECTORDB)

st.set_page_config(page_title="Asset Classifier", layout="wide")

# ---------- Startup ----------
@st.cache_resource
def bootstrap():
    VECTORDB.load_or_init()
    RAG.init_db()
    return True

bootstrap()

st.title("🔎 Asset Classification — Query • Verify • Feedback")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K retrieved", min_value=3, max_value=20, value=8, step=1)
    llm_threshold = st.slider("LLM trigger if sim < ", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    st.caption("Above: when top similarity is lower than this, LLM RAG is used.")
    st.divider()
    st.subheader("Paths")
    st.text(f"SQLite: {SETTINGS.sqlite_path}")
    st.text(f"FAISS:  {SETTINGS.faiss_index_path}")

tab_query, tab_feedback, tab_recent = st.tabs(["🔍 Query & Verify", "✍️ Submit Feedback", "🗂️ Recent Submissions"])

# ---------- Tab 1: Query & Verify ----------
with tab_query:
    st.subheader("1) Enter an equipment description")
    query = st.text_input("Equipment description", placeholder="e.g., PIG LAUNCHER 4IN BALL VALVE")

    colA, colB = st.columns([1,1])
    with colA:
        desc1_opt = st.text_input("Description1 (optional)", placeholder="Additional context")
    with colB:
        show_raw_norm = st.checkbox("Show normalized text", value=False)

    run_btn = st.button("Run Classification", type="primary")

    
    if "last_res" not in st.session_state:
        st.session_state.last_res = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""


    if run_btn and query.strip():
        with st.spinner("Retrieving and classifying..."):
            # Run classification
            res = RAG.classify(query, top_k=top_k, llm_when_sim_below=llm_threshold)

        
        st.session_state.last_res = res
        st.session_state.last_query = query
    
    
    if st.session_state.last_res:
        res = st.session_state.last_res
        query = st.session_state.last_query 


        st.markdown("### Result")
        left, right = st.columns([1,1])

        with left:
            st.json({
                "mode": res.get("mode"),
                "prediction": res.get("prediction"),
                "rationale": res.get("rationale"),
                "retrieved": res.get("retrieved")
            })

            if show_raw_norm:
                st.code(RAG.normalize_text(query), language="text")

        with right:
            st.markdown("#### Retrieved examples (top-k)")
            retrieved = res.get("retrieved", [])
            if retrieved:
                st.dataframe(pd.DataFrame(retrieved), width='stretch')
            else:
                st.info("No retrieved examples found.")

        pred = res.get("prediction", {})
        st.markdown("### 2) Verify / Adjust classification before saving")

        # Offer dropdowns seeded from retrieved classes and free text overrides
        retrieved_rows, _ = RAG.retrieve(query, top_k=top_k)
        l1_options = list({r['l1'] for r in retrieved_rows}) or []
        l2_options = list({r['l2'] for r in retrieved_rows}) or []
        l3_options = list({r['l3'] for r in retrieved_rows}) or []

        c1, c2, c3 = st.columns(3)
        with c1:
            level1 = st.selectbox("Level 1", options=[pred.get("level1","")] + l1_options, index=0)
            level1 = st.text_input("Or enter Level 1 manually", value=level1)
        with c2:
            level2 = st.selectbox("Level 2", options=[pred.get("level2","")] + l2_options, index=0)
            level2 = st.text_input("Or enter Level 2 manually", value=level2)
        with c3:
            level3 = st.selectbox("Level 3", options=[pred.get("level3","")] + l3_options, index=0)
            level3 = st.text_input("Or enter Level 3 manually", value=level3)

        st.markdown("### 3) (Optional) Add metadata")
        m1, m2 = st.columns(2)
        with m1:
            manufacturer = st.text_input("Manufacturer", value=st.session_state.get("manufacturer",""), key="manu_text")
        with m2:
            model_number = st.text_input("Model number", value=st.session_state.get("model_text",""), key="model_text")

        if st.button("✅ Submit feedback", type="primary", width='stretch', key="submit_fb"):
            # st.write("innnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
            try:
                print("Submitting feedback for:", query, level1, level2, level3, manufacturer, model_number, desc1_opt)
                fb = RAG.submit_feedback(
                    equip_description=query,
                    description1=desc1_opt or "",
                    manufacturer=manufacturer,
                    model_number=model_number,
                    level1=level1,
                    level2=level2,
                    level3=level3,
                    source="feedback"
                )
                st.success(f"Saved! Row ID: {fb.get('row_id')}")
            except Exception as e:
                st.error(f"Failed to submit feedback: {e}")

# ---------- Tab 2: Direct Feedback Entry ----------
with tab_feedback:
    st.subheader("Submit / Correct a classification")
    f_query = st.text_input("Equipment description*", key="fb_q")
    f_desc1 = st.text_input("Description1", key="fb_d1")
    f_manuf = st.text_input("Manufacturer", key="fb_manu")
    f_model = st.text_input("Model number", key="fb_model")

    f1, f2, f3 = st.columns(3)
    with f1:
        f_l1 = st.text_input("Level 1*")
    with f2:
        f_l2 = st.text_input("Level 2*")
    with f3:
        f_l3 = st.text_input("Level 3*")

    if st.button("💾 Save feedback", type="primary"):
        if not f_query.strip() or not f_l1.strip() or not f_l2.strip() or not f_l3.strip():
            st.warning("Please fill required fields (*)")
        else:
            try:
                fb = RAG.submit_feedback(
                    equip_description=f_query,
                    description1=f_desc1,
                    manufacturer=f_manuf,
                    model_number=f_model,
                    level1=f_l1,
                    level2=f_l2,
                    level3=f_l3,
                    source="feedback"
                )
                st.success(f"Saved! Row ID: {fb.get('row_id')}")
            except Exception as e:
                st.error(f"Error saving feedback: {e}")

# ---------- Tab 3: Browse Recent Submissions ----------
with tab_recent:
    st.subheader("Recent entries in SQLite")
    limit = st.slider("Show last N rows", min_value=10, max_value=500, value=100, step=10)

    def fetch_recent(n=100):
        with RAG.open_db() as conn:
            rows = conn.execute(
                "SELECT id, equip_description, description1, manufacturer, model_number, l1, l2, l3, source, created_at "
                "FROM asset_examples ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
            cols = ["id","equip_description","description1","manufacturer","model_number","l1","l2","l3","source","created_at"]
            return pd.DataFrame(rows, columns=cols)

    df = fetch_recent(limit)
    st.dataframe(df, width='stretch', height=500)

    csv = df.to_csv(index=False)
    st.download_button("⬇️ Export CSV", data=csv, file_name=f"asset_examples_{datetime.now().date()}.csv", mime="text/csv")

