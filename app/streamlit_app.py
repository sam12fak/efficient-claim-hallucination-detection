"""
Streamlit Application for Efficient Claim-Level Hallucination Detection & Interactive RAG
Combines Local Ollama LLMs (qwen2.5:3b), Dense Vector Semantic Search (all-MiniLM-L6-v2),
and Calibrated Hybrid Tree Classifiers for Real-Time Faithfulness Verification.
"""

import sys
import os
import time
import shutil
import tempfile
import pandas as pd
import torch
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

from src.predict import predict_hallucinations, load_detector_model
from src.claim_extraction import get_spacy_nlp
from src.semantic_features import get_embedding_model
from src.rag_pipeline import get_rag_pipeline, RAGHallucinationPipeline

st.set_page_config(
    page_title="NeuralFact // RAG & Hallucination Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, Netflix-inspired dark/light theme aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .timing-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .timing-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #2563EB;
        font-family: monospace;
    }
    .timing-lbl {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }
    .score-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .score-val {
        font-size: 2.0rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .score-val.good { color: #16A34A; }
    .score-val.warn { color: #D97706; }
    .score-val.bad { color: #DC2626; }
    .claim-card-supported {
        background-color: #F0FDF4;
        border-left: 5px solid #22C55E;
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .claim-card-hallu {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        border-radius: 8px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .badge-supported {
        display: inline-block;
        background-color: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.82rem;
    }
    .badge-hallu {
        display: inline-block;
        background-color: #FEE2E2;
        color: #B91C1C;
        font-weight: 700;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.82rem;
    }
    .source-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
    }
    .ollama-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .ollama-online {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .ollama-offline {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_system():
    """Initializes and caches models and RAG pipeline once during startup."""
    nlp = get_spacy_nlp()
    emb = get_embedding_model()
    model, feature_names = load_detector_model()
    rag_pipe = get_rag_pipeline()
    return nlp, emb, model, feature_names, rag_pipe


nlp, emb_model, detector_model, feature_names, rag_pipeline = init_system()

# Check Ollama connection status
ollama_online = rag_pipeline.ollama.is_available()
available_models = rag_pipeline.ollama.list_models() if ollama_online else ["qwen2.5:3b (Fallback)"]

# -------------------------------------------------------------
# Sidebar: System Config & Presets
# -------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Control")

    # Ollama Health Status Pill
    if ollama_online:
        st.markdown(
            '<div class="ollama-pill ollama-online">🟢 Ollama LLM Service Active (Port 11434)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="ollama-pill ollama-offline">🔴 Ollama Offline — Fallback Generator Enabled</div>',
            unsafe_allow_html=True
        )

    st.caption(f"**Indexed RAG Documents:** `{len(rag_pipeline.kb.chunks)} chunks`")

    # Hardware detection
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU Execution"
    st.caption(f"**Compute Device:** `{device_name}`")
    st.caption("**Embeddings:** `all-MiniLM-L6-v2` (384-d)")
    st.caption("**Detector:** `Calibrated XGBoost / Hybrid Ensemble`")

    st.divider()
    st.subheader("🤖 Local LLM Engine")
    selected_llm = st.selectbox("Ollama Model:", available_models, index=0)

    st.divider()
    st.subheader("🎯 Risk Thresholds")
    threshold = st.slider("Hallucination Risk Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    high_risk_threshold = st.slider("High Risk Flag Threshold", min_value=0.50, max_value=0.95, value=0.75, step=0.05)

    st.divider()
    st.subheader("🛡️ Detector Classifier")
    model_choice = st.selectbox(
        "Classifier Model:",
        [
            "⚡ Hybrid XGBoost (Calibrated)",
            "🌲 Hybrid Random Forest (Calibrated)",
            "🚀 Grounded Calibrated Ensemble"
        ]
    )

    st.divider()
    with st.expander("📊 Technical Benchmark Summary"):
        st.markdown("""
        **Evaluated on HaluEval (4,228 claims):**
        - **Model F1 Score:** 0.7838 (RF) / 0.7798 (XGBoost)
        - **ROC-AUC Score:** 0.8550
        - **Inference Latency:** 0.001 ms / claim
        - **Throughput:** > 980,000 claims/sec
        - **Zero GPU or External API dependency.**
        """)

# -------------------------------------------------------------
# Main Application Content
# -------------------------------------------------------------
st.markdown('<div class="main-title">🔍 NeuralFact // RAG & Hallucination Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">End-to-End Retrieval-Augmented Generation with Local Ollama LLM and Sub-Millisecond Hallucination Guardrails.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "⚡ Interactive RAG with Ollama",
    "🔬 Standalone Claim Detector",
    "📚 Course Knowledge Base & Documents"
])

# =============================================================
# TAB 1: INTERACTIVE RAG WITH OLLAMA
# =============================================================
with tab1:
    st.markdown("### 🎓 Computational Linguistics Course Assistant")
    st.caption("Ask questions about the syllabus, lecture slides (PPTX), and course unit notes (PDF). The system retrieves evidence, queries the local Ollama LLM, and immediately fact-checks every generated claim.")

    # Quick sample queries
    st.markdown("**Sample Course Queries (Click to populate):**")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)

    preset_rag_query = None
    with q_col1:
        if st.button("📋 Grading & Syllabus", use_container_width=True):
            preset_rag_query = "What are the grading percentages, assignments, and late policy in the course syllabus?"
    with q_col2:
        if st.button("📦 Products of CL (Unit 3)", use_container_width=True):
            preset_rag_query = "What are the main products and applications of Computational Linguistics in Unit 3?"
    with q_col3:
        if st.button("🌳 CKY & Parsing (Unit 4)", use_container_width=True):
            preset_rag_query = "Explain how linguistic models and syntax parsing transform text to meaning."
    with q_col4:
        if st.button("🔤 Morphology & Tokenizers", use_container_width=True):
            preset_rag_query = "What is the difference between Porter stemming, lemmatization, and subword tokenization?"

    # Query Input Area
    if "rag_query_text" not in st.session_state:
        st.session_state.rag_query_text = "What are the grading percentages, assignments, and late policy in the course syllabus?"

    if preset_rag_query:
        st.session_state.rag_query_text = preset_rag_query

    rag_user_query = st.text_input(
        "Enter your question for the Course RAG Agent:",
        value=st.session_state.rag_query_text,
        placeholder="e.g. What are the products of Computational Linguistics?"
    )

    r_ctrl1, r_ctrl2, r_ctrl3 = st.columns([1, 1, 1.2])
    with r_ctrl1:
        top_k_val = st.slider("Retrieved Context Chunks (Top-K):", min_value=1, max_value=6, value=3)
    with r_ctrl2:
        stress_test_mode = st.checkbox(
            "⚠️ Hallucination Stress-Test",
            value=False,
            help="Instructs the LLM to speculate beyond the retrieved notes. Great for observing the detector catching hallucinations live!"
        )
    with r_ctrl3:
        st.write("")
        st.write("")
        run_rag_btn = st.button("🚀 Ask RAG & Verify Claims", type="primary", use_container_width=True)

    if run_rag_btn:
        if not rag_user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Executing RAG Pipeline: Retrieving Chunks ➔ Ollama Inference ➔ Claim Hallucination Detection..."):
                active_model_key = "hybrid_xgboost" if "XGBoost" in model_choice else "hybrid_random_forest"
                rag_res = rag_pipeline.run_query(
                    query=rag_user_query,
                    top_k=top_k_val,
                    model_name=selected_llm,
                    stress_test=stress_test_mode,
                    detector_model_type=active_model_key
                )

            # ---------------------------------------------------------
            # 1. Performance & Latency Dashboard
            # ---------------------------------------------------------
            st.markdown("#### ⏱️ Real-Time Pipeline Latency & Telemetry")
            t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
            with t_col1:
                st.markdown(f"""
                <div class="timing-card">
                    <div class="timing-val">{rag_res.timing['retrieval_ms']} ms</div>
                    <div class="timing-lbl">Vector Retrieval</div>
                </div>
                """, unsafe_allow_html=True)
            with t_col2:
                st.markdown(f"""
                <div class="timing-card">
                    <div class="timing-val">{rag_res.timing['llm_ms']} ms</div>
                    <div class="timing-lbl">Ollama Inference</div>
                </div>
                """, unsafe_allow_html=True)
            with t_col3:
                st.markdown(f"""
                <div class="timing-card">
                    <div class="timing-val">{rag_res.timing['tokens_per_sec']} t/s</div>
                    <div class="timing-lbl">Generation Speed</div>
                </div>
                """, unsafe_allow_html=True)
            with t_col4:
                st.markdown(f"""
                <div class="timing-card">
                    <div class="timing-val">{rag_res.timing['detection_ms']} ms</div>
                    <div class="timing-lbl">Fact Verification</div>
                </div>
                """, unsafe_allow_html=True)
            with t_col5:
                st.markdown(f"""
                <div class="timing-card" style="border-color: #93C5FD; background: #EFF6FF;">
                    <div class="timing-val" style="color: #1D4ED8;">{rag_res.timing['total_ms']} ms</div>
                    <div class="timing-lbl">Total End-to-End</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

            # ---------------------------------------------------------
            # 2. Factuality & Faithfulness Assessment
            # ---------------------------------------------------------
            s_col1, s_col2, s_col3, s_col4 = st.columns(4)

            faith_score = rag_res.faithfulness_score
            risk_score = rag_res.overall_risk_percent
            faith_class = "good" if faith_score >= 70 else ("warn" if faith_score >= 40 else "bad")
            risk_class = "bad" if risk_score >= 50 else ("warn" if risk_score >= 25 else "good")

            with s_col1:
                st.markdown(f"""
                <div class="score-card">
                    <div class="score-val {faith_class}">{faith_score}%</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: #475569;">FAITHFULNESS INDEX</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Corroboration against syllabus</div>
                </div>
                """, unsafe_allow_html=True)

            with s_col2:
                st.markdown(f"""
                <div class="score-card">
                    <div class="score-val {risk_class}">{risk_score}%</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: #475569;">HALLUCINATION RISK</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Calibrated model probability</div>
                </div>
                """, unsafe_allow_html=True)

            with s_col3:
                st.markdown(f"""
                <div class="score-card">
                    <div class="score-val" style="color: #3B82F6;">{rag_res.supported_claims_count} / {rag_res.total_claims}</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: #475569;">SUPPORTED CLAIMS</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Verified factual assertions</div>
                </div>
                """, unsafe_allow_html=True)

            with s_col4:
                status_color = "#16A34A" if rag_res.status == "Supported" else ("#DC2626" if rag_res.status == "Hallucinated" else "#D97706")
                st.markdown(f"""
                <div class="score-card">
                    <div class="score-val" style="color: {status_color}; font-size: 1.55rem; padding-top: 5px;">{rag_res.status}</div>
                    <div style="font-size: 0.85rem; font-weight: 700; color: #475569;">OVERALL VERDICT</div>
                    <div style="font-size: 0.75rem; color: #64748B;">Threshold: {threshold*100:.0f}% risk</div>
                </div>
                """, unsafe_allow_html=True)

            # Progress Indicator
            st.write("")
            progress_val = min(1.0, max(0.0, faith_score / 100.0))
            st.progress(progress_val, text=f"Faithfulness Score: {faith_score}%")

            # ---------------------------------------------------------
            # 3. LLM Generated Answer
            # ---------------------------------------------------------
            st.markdown("#### 💬 Generated Response (Ollama Local LLM)")
            st.info(f"**Model:** `{rag_res.llm_model}` | **Mode:** `{'⚠️ Speculative Stress-Test' if stress_test_mode else 'Factually Grounded'}`\n\n{rag_res.llm_answer}")

            # ---------------------------------------------------------
            # 4. Claim-Level Breakdown
            # ---------------------------------------------------------
            st.markdown("#### 📑 Claim-Level Factuality Breakdown")
            st.caption("Each sentence generated by the LLM is parsed into discrete claims, cross-checked against retrieved source context, and scored for semantic distance, entity alignment, and numerical precision.")

            claims = rag_res.hallucination_analysis.get("claims", [])
            if not claims:
                st.write("No distinct factual claims extracted.")
            else:
                for c in claims:
                    is_hallu = c.get("prediction") == "Hallucination" or c.get("prediction") == "Likely Hallucination"
                    card_cls = "claim-card-hallu" if is_hallu else "claim-card-supported"
                    badge_cls = "badge-hallu" if is_hallu else "badge-supported"
                    icon = "⚠️" if is_hallu else "✅"

                    st.markdown(f"""
                    <div class="{card_cls}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-weight: 700; color: #0F172A;">Claim #{c['claim_id']}</span>
                            <span class="{badge_cls}">{icon} {c['prediction']} &bull; Risk: {c['hallucination_risk']}%</span>
                        </div>
                        <div style="font-size: 1.02rem; color: #1E293B; margin-bottom: 0.6rem; font-style: italic;">
                            "{c['claim_text']}"
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    c_col1, c_col2 = st.columns([2, 1])
                    with c_col1:
                        # Grounding entities
                        grounding = c.get("grounding", {})
                        supp_ents = grounding.get("supported_entities", [])
                        unsupp_ents = grounding.get("unsupported_entities", [])
                        if supp_ents:
                            st.caption(f"✅ **Corroborated Entities in Notes:** `{'`, `'.join(supp_ents)}`")
                        if unsupp_ents:
                            st.caption(f"⚠️ **Uncorroborated / Invented Entities:** `{'`, `'.join(unsupp_ents)}`")

                        st.markdown("**Model Signals:**")
                        for sig in c.get("model_signals", []):
                            st.markdown(f"- {sig}")

                    with c_col2:
                        with st.expander(f"Inspect Claim #{c['claim_id']} Feature Vector"):
                            feat_df = pd.DataFrame(
                                list(c["features"].items()),
                                columns=["Feature Dimension", "Value"]
                            )
                            st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # ---------------------------------------------------------
            # 5. Retrieved Grounding Context Sources
            # ---------------------------------------------------------
            with st.expander(f"📚 View Retrieved Course Context Chunks ({len(rag_res.retrieved_chunks)} Chunks)"):
                for idx, ch in enumerate(rag_res.retrieved_chunks, start=1):
                    file_ext_badge = ch["source_type"].upper()
                    st.markdown(f"""
                    <div class="source-box">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                            <strong>Chunk #{idx}: <code>{ch['source_file']}</code> (Page/Slide {ch['page_or_slide']})</strong>
                            <span style="color: #2563EB; font-weight: 600;">Similarity: {ch['similarity_score']:.4f}</span>
                        </div>
                        <div style="color: #334155;">{ch['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)


# =============================================================
# TAB 2: STANDALONE CLAIM DETECTOR
# =============================================================
with tab2:
    st.markdown("### 🔬 Manual Claim Verification Console")
    st.caption("Provide an arbitrary user question, reference context, and LLM generated response to evaluate claim-level hallucination risk without running RAG.")

    preset_choice = st.selectbox(
        "Select a preconfigured scenario:",
        [
            "Custom Input",
            "1. Factual Answer (Expected Low Risk)",
            "2. Clearly Hallucinated (Expected High Risk)",
            "3. Mixed Multi-Claim Answer",
            "4. Long Historical Narrative"
        ]
    )

    preset_data = {
        "1. Factual Answer (Expected Low Risk)": {
            "question": "Who discovered penicillin?",
            "answer": "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London.",
            "context": "Sir Alexander Fleming was a Scottish physician and microbiologist who discovered penicillin in 1928 at St. Mary's Hospital in London."
        },
        "2. Clearly Hallucinated (Expected High Risk)": {
            "question": "Who discovered penicillin?",
            "answer": "Marie Curie discovered penicillin in 1928 while conducting radioactivity research in Paris.",
            "context": "Sir Alexander Fleming was a Scottish physician and microbiologist who discovered penicillin in 1928."
        },
        "3. Mixed Multi-Claim Answer": {
            "question": "Who founded SpaceX and what was its background?",
            "answer": "Elon Musk founded SpaceX in 2002. SpaceX is headquartered in Texas. SpaceX was originally called NASA Technologies.",
            "context": "SpaceX was founded in 2002 by Elon Musk with the goal of reducing space transportation costs."
        },
        "4. Long Historical Narrative": {
            "question": "Tell me about the Apollo 11 moon mission.",
            "answer": "The Apollo 11 mission was launched on July 16, 1969. Neil Armstrong was the mission commander. The lunar module landed on Mars after three days of travel. Armstrong became the first human to step onto the surface.",
            "context": "Apollo 11 was the American spaceflight that first landed humans on the Moon on July 20, 1969. Neil Armstrong was the mission commander who first stepped on the surface."
        }
    }

    default_q = ""
    default_a = ""
    default_ctx = ""
    if preset_choice in preset_data:
        default_q = preset_data[preset_choice]["question"]
        default_a = preset_data[preset_choice]["answer"]
        default_ctx = preset_data[preset_choice]["context"]

    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        q_inp = st.text_input("User Question:", value=default_q, key="manual_q")
        ctx_inp = st.text_area("Reference Context (or leave blank):", value=default_ctx, height=110, key="manual_ctx")
    with sc_col2:
        ans_inp = st.text_area("LLM Generated Answer:", value=default_a, height=195, key="manual_ans")

    if st.button("🚀 Evaluate Answer", type="primary", use_container_width=True):
        if not q_inp.strip() or not ans_inp.strip():
            st.warning("Please provide Question and Answer.")
        else:
            with st.spinner("Analyzing semantic distance, entity alignment, and linguistic stance..."):
                active_model = None
                if "XGBoost" in model_choice:
                    active_model, _ = load_detector_model("models/hybrid_xgboost.joblib")
                elif "Random Forest" in model_choice:
                    active_model, _ = load_detector_model("models/best_model.joblib")

                results = predict_hallucinations(
                    question=q_inp,
                    answer=ans_inp,
                    context=ctx_inp,
                    threshold=threshold,
                    high_risk_threshold=high_risk_threshold,
                    model=active_model,
                    nlp=nlp,
                    emb_model=emb_model
                )

            st.divider()
            om_col1, om_col2, om_col3, om_col4 = st.columns(4)
            with om_col1:
                st.metric("Overall Risk", f"{results['overall_risk_percent']}%")
            with om_col2:
                st.metric("Peak Claim Risk", f"{results['max_risk_percent']}%")
            with om_col3:
                st.metric("Total Claims", results["total_claims"])
            with om_col4:
                st.metric("High-Risk Claims", f"{results['high_risk_claim_count']} ({results['high_risk_percentage']}%)")

            for cl in results["claims"]:
                is_h = cl["prediction"] in ("Hallucination", "Likely Hallucination")
                st.markdown(f"""
                <div class="{'claim-card-hallu' if is_h else 'claim-card-supported'}">
                    <b>Claim {cl['claim_id']}</b>: {cl['claim_text']}
                    <br><span class="{'badge-hallu' if is_h else 'badge-supported'}">{cl['prediction']} (Risk: {cl['hallucination_risk']}%)</span>
                </div>
                """, unsafe_allow_html=True)


# =============================================================
# TAB 3: COURSE KNOWLEDGE BASE & DOCUMENT UPLOADER
# =============================================================
with tab3:
    st.markdown("### 📚 Knowledge Base Documents")
    st.caption("Manage documents indexed in the RAG vector store. You can inspect indexed files or upload additional Computational Linguistics PPTX slides, PDFs, or notes.")

    docs_folder = "data/rag_docs"
    existing_files = [f for f in os.listdir(docs_folder) if os.path.isfile(os.path.join(docs_folder, f))]

    file_info = []
    for f in existing_files:
        p = os.path.join(docs_folder, f)
        size_kb = round(os.path.getsize(p) / 1024, 1)
        ext = os.path.splitext(f)[1].upper().replace(".", "")
        file_info.append({"Filename": f, "Type": ext, "Size (KB)": size_kb})

    st.dataframe(pd.DataFrame(file_info), use_container_width=True, hide_index=True)

    st.markdown("#### 📤 Upload Additional Slides or Notes")
    uploaded_files = st.file_uploader(
        "Upload PPTX slides, PDF textbook chapters, or Markdown notes:",
        type=["pdf", "pptx", "ppt", "txt", "md"],
        accept_multiple_files=True
    )

    u_col1, u_col2 = st.columns([1, 1])
    with u_col1:
        if uploaded_files:
            if st.button("💾 Save and Re-Index Documents", type="primary", use_container_width=True):
                with st.spinner("Processing uploads and computing dense embeddings..."):
                    for uf in uploaded_files:
                        dest_path = os.path.join(docs_folder, uf.name)
                        with open(dest_path, "wb") as out_f:
                            out_f.write(uf.getbuffer())
                    new_count = rag_pipeline.kb.build_or_load_index(force_rebuild=True)
                st.success(f"Successfully uploaded and indexed! Knowledge base now contains {new_count} chunks.")
                st.rerun()

    with u_col2:
        if st.button("🔄 Force Rebuild Entire Index", use_container_width=True):
            with st.spinner("Re-embedding all files in data/rag_docs..."):
                new_count = rag_pipeline.kb.build_or_load_index(force_rebuild=True)
            st.success(f"Re-indexed {new_count} chunks from {len(existing_files)} files!")
            st.rerun()

# Footer
st.divider()
st.caption("NeuralFact // Research Engine for Real-Time LLM Factuality & Claim-Level Hallucination Detection.")
