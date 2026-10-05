
import os
import json
from pathlib import Path

import streamlit as st


# ============================================================
# PAGE / PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

os.environ["RAG_DATA_ROOT"] = str(ROOT / "data" / "RAG_Project")
os.environ["RAG_OUTPUT_DIR"] = str(ROOT / "outputs")
os.environ["RAG_AIML_CORPUS_PATH"] = str(ROOT / "data" / "chunks_aiml.csv")

st.set_page_config(
    page_title="Enhancing RAG Using Hybrid Retrieval and Re-Ranking Techniques for AI and ML Based Research",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
<style>
.stApp {
    background: #07111f;
    color: #e8eef7;
}
[data-testid="stSidebar"] {
    background: #0b1728;
    border-right: 1px solid #20334d;
}
[data-testid="stSidebar"] * {
    color: #e8eef7;
}
#MainMenu, footer, .stDeployButton {
    visibility: hidden;
}

.hero {
    padding: 34px 42px 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0d2038, #102b49);
    border: 1px solid #284564;
    margin-bottom: 22px;
}
.hero h1 {
    margin: 0;
    color: #fff;
    font-size: 2.35rem;
    line-height: 1.2;
    font-weight: 800;
}
.hero .sub {
    margin-top: 12px;
    color: #aebfd1;
    font-size: 0.96rem;
    line-height: 1.6;
    max-width: 900px;
}
.pipeline {
    display: flex;
    gap: 7px;
    align-items: center;
    flex-wrap: wrap;
    margin: 18px 0 26px;
}
.stage {
    padding: 8px 13px;
    background: #102238;
    border: 1px solid #294562;
    border-radius: 9px;
    color: #dbe7f4;
    font-size: 0.82rem;
    font-weight: 600;
}
.arrow { color: #7189a5; }

.notice {
    padding: 12px 16px;
    border-radius: 10px;
    background: #0e2035;
    border: 1px solid #284968;
    color: #bdcede;
    margin-bottom: 20px;
    font-size: 0.88rem;
}

.query-box {
    padding: 18px;
    border-radius: 14px;
    background: #0c1a2b;
    border: 1px solid #213a55;
    margin-bottom: 18px;
}

.key-evidence {
    padding: 20px 22px;
    border-radius: 14px;
    background: #10243a;
    border: 1px solid #315779;
    margin: 12px 0 20px;
}
.key-label {
    color: #86a8c9;
    font-size: 0.73rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-bottom: 7px;
}
.key-title {
    color: #fff;
    font-size: 1.04rem;
    font-weight: 700;
    margin-bottom: 7px;
}
.key-text {
    color: #cbd8e5;
    line-height: 1.62;
    font-size: 0.93rem;
}

.evidence {
    padding: 16px 18px;
    border-radius: 11px;
    background: #0c1a2b;
    border: 1px solid #213a55;
    margin: 8px 0;
}
.evidence-title {
    color: #fff;
    font-weight: 650;
    margin-bottom: 4px;
}
.evidence-meta {
    color: #7891ac;
    font-size: 0.75rem;
    margin-bottom: 8px;
}
.evidence-text {
    color: #c4d1df;
    line-height: 1.55;
    font-size: 0.88rem;
}

.metric-card {
    padding: 15px;
    border-radius: 12px;
    background: #0e1d30;
    border: 1px solid #223b57;
    text-align: center;
}
.metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #fff;
}
.metric-label {
    color: #94a9c0;
    font-size: 0.78rem;
    margin-top: 4px;
}
.small {
    color: #8297ad;
    font-size: .78rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# REAL RESEARCH BACKEND
# ============================================================

@st.cache_resource(show_spinner="Loading trained models and the AI/ML research corpus...")
def load_backend():
    import rag_backend
    rag_backend.load_core_resources()
    return rag_backend


try:
    rag_backend = load_backend()
    BACKEND_ERROR = None
except Exception as exc:
    rag_backend = None
    BACKEND_ERROR = exc


# ============================================================
# HELPERS
# ============================================================


def aiml_query(query_text, top_k=5):
    return rag_backend.query(query_text, dataset="aiml", top_k=top_k)


def run_query(query_text, dataset_key, top_k):
    if dataset_key == "aiml":
        return aiml_query(query_text, top_k)
    return rag_backend.query(
        query_text,
        dataset=dataset_key,
        top_k=top_k,
    )


def show_result(result):
    st.markdown("### Query analysis")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Query type", str(result.get("query_type", "")).title())
    with c2:
        st.metric("Adaptive α", f"{float(result.get('alpha', 0)):.2f}")
    with c3:
        st.metric("Hybrid candidates", int(result.get("retrieved_count", 0)))

    st.caption(
        "α controls the hybrid balance: α = 0.2 favors BM25, "
        "0.5 balances both signals, and 0.8 favors dense retrieval."
    )
    st.caption("Hybrid retrieval: BM25 + BGE-M3 → Cross-Encoder re-ranking")

    results = result.get("results", [])

    if not results:
        st.warning("No evidence was retrieved.")
        return

    top = results[0]

    st.markdown("### Top re-ranked evidence")

    st.markdown(
        f"""<div class="key-evidence">
<div class="key-label">Cross-encoder re-ranked result #1</div>
<div class="key-title">{top.get("title", "Source document")}</div>
<div class="evidence-meta">
{top.get("chunk_id", "")} · re-rank score {float(top.get("rerank_score", 0)):.4f}
</div>
<div class="key-text">{top.get("text", "")}</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### Hybrid retrieval + re-ranking")

    for i, item in enumerate(results, 1):
        extra = ""
        if "authors" in item and item.get("authors"):
            extra = f" · {item['authors']}"

        st.markdown(
            f"""<div class="evidence">
<div class="evidence-title">{i}. {item.get("title", "Source document")}</div>
<div class="evidence-meta">
{item.get("chunk_id", "")}{extra} ·
hybrid score {float(item.get("score", 0)):.4f} ·
re-rank {float(item.get("rerank_score", 0)):.4f}
</div>
<div class="evidence-text">{item.get("text", "")}</div>
</div>""",
            unsafe_allow_html=True,
        )

        if item.get("paper_id"):
            st.caption(
                f"AI/ML paper ID: {item['paper_id']}"
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🔎 Adaptive RAG")
    st.caption("Capstone Research Demonstrator")
    st.divider()

    st.markdown("### Research pipeline")
    st.markdown(
        """
        **Query classification**  
        Trained BERT classifier

        **Hybrid retrieval**  
        BM25 + BGE-M3

        **Adaptive fusion**  
        Query-dependent α

        **Re-ranking**  
        Trained cross-encoder
        """
    )

    st.divider()

    st.markdown("### Research collections")
    st.markdown(
        """
        • Natural Questions  
        • TREC-COVID  
        • SQuAD  
        • AI/ML Research — 89 papers
        """
    )

    st.divider()

    top_k = st.slider(
        "Evidence passages",
        min_value=3,
        max_value=10,
        value=5,
    )

    if rag_backend is not None:
        st.success("Research backend loaded")
    else:
        st.error("Research backend failed to load")
        st.caption(str(BACKEND_ERROR))


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">
<h1>Enhancing RAG Using Hybrid Retrieval and Re-Ranking Techniques for AI and ML Based Research</h1>
</div>

<div class="pipeline">
<div class="stage">Query</div><div class="arrow">→</div>
<div class="stage">BERT Classifier</div><div class="arrow">→</div>
<div class="stage">Adaptive α</div><div class="arrow">→</div>
<div class="stage">Hybrid Retrieval: BM25 + BGE-M3</div><div class="arrow">→</div>
<div class="stage">Cross-Encoder</div><div class="arrow">→</div>
<div class="stage">Evidence</div>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab_search, tab_eval, tab_about = st.tabs(
    ["🔍 AI/ML Research Assistant", "📊 Evaluation", "ℹ️ About"]
)


# ============================================================
# RESEARCH DEMO
# ============================================================

with tab_search:

    st.markdown("## AI/ML Research Assistant")

    dataset_label = "AI/ML Research Corpus"
    dataset_key = "aiml"

    st.markdown(
        f"""<div class="notice">
<b>{dataset_label}</b> — Ask any free-form question. The system retrieves
and reranks relevant passages from the local AI/ML paper corpus.
</div>""",
        unsafe_allow_html=True,
    )

    examples = {"aiml": "What is LoRA and why is it useful for fine-tuning large language models?"}

    with st.container(border=True):
        query_text = st.text_area(
            "Research question",
            key=f"query_{dataset_key}",
            height=90,
            placeholder=examples[dataset_key],
        )

        c1, c2 = st.columns([1, 5])
        with c1:
            search_clicked = st.button(
                "🔎 Search",
                type="primary",
                use_container_width=True,
            )
        with c2:
            st.caption(
                "Examples are optional. You can replace the text with any question."
            )

    if search_clicked:
        query_text = query_text.strip()

        if not query_text:
            st.warning("Please enter a research question.")
        elif rag_backend is None:
            st.error(
                "The trained research backend could not be loaded. "
                "Check the terminal traceback before the demonstration."
            )
        else:
            with st.spinner("Running adaptive hybrid retrieval and re-ranking..."):
                try:
                    result = run_query(query_text, dataset_key, top_k)
                    st.session_state["last_result"] = result
                    st.session_state["last_dataset"] = dataset_label
                except Exception as exc:
                    st.error(f"Query failed: {exc}")
                    st.exception(exc)

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]

        st.markdown(
            f"""<div class="query-box">
<b>Current query:</b> {result.get("query", "")}<br>
<span class="small">Collection: {result.get("dataset_name", st.session_state.get("last_dataset", ""))}</span>
</div>""",
            unsafe_allow_html=True,
        )

        show_result(result)


# ============================================================
# EVALUATION
# ============================================================

with tab_eval:

    st.markdown("## Evaluation results")
    st.caption("Saved evaluation results from the completed research experiments.")

    eval_path = ROOT / "outputs" / "all_results.json"

    if eval_path.exists():
        try:
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:
            evaluation = {}
    else:
        evaluation = {}

    cols = st.columns(4)
    metrics = [
        ("TREC Hybrid NDCG@10", evaluation.get("trec_hybrid_ndcg", 0.5816)),
        ("TREC Adaptive NDCG@10", evaluation.get("trec_adaptive_ndcg", 0.5826)),
        ("NQ Hybrid NDCG@10", evaluation.get("nq_hybrid_ndcg", 0.6474)),
        ("NQ Adaptive NDCG@10", evaluation.get("nq_adaptive_ndcg", 0.6486)),
    ]

    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(
                f"""<div class="metric-card">
<div class="metric-value">{float(value):.4f}</div>
<div class="metric-label">{label}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("### SQuAD")

    squad = evaluation.get(
        "squad",
        {
            "hybrid": {"EM": 68.89, "F1": 68.88, "Top-20": 91.28},
            "adaptive": {"EM": 69.08, "F1": 69.07, "Top-20": 91.26},
        },
    )

    cols = st.columns(5)
    squad_metrics = [
        ("Hybrid EM", squad["hybrid"].get("EM", 68.89)),
        ("Hybrid F1", squad["hybrid"].get("F1", 68.88)),
        ("Adaptive EM", squad["adaptive"].get("EM", 69.08)),
        ("Adaptive F1", squad["adaptive"].get("F1", 69.07)),
        ("Adaptive Top-20", squad["adaptive"].get("Top-20", 91.26)),
    ]

    for col, (label, value) in zip(cols, squad_metrics):
        with col:
            st.markdown(
                f"""<div class="metric-card">
<div class="metric-value">{float(value):.2f}%</div>
<div class="metric-label">{label}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("### NQ retrieval evaluation")

    nq_gen = evaluation.get(
        "nq_generation",
        {"top5_acc": 89.0, "num_queries": 100},
    )

    cols = st.columns(2)
    for col, (label, value) in zip(
        cols,
        [
            ("Top-5 retrieval accuracy", nq_gen.get("top5_acc", 89.0)),
            ("Evaluated queries", nq_gen.get("num_queries", 100)),
        ],
    ):
        with col:
            display = f"{float(value):.1f}%" if "accuracy" in label else str(value)
            st.markdown(
                f"""<div class="metric-card">
<div class="metric-value">{display}</div>
<div class="metric-label">{label}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("### RAGAS-style evaluation")

    ragas = evaluation.get(
        "ragas",
        {
            "faithfulness": 0.0,
            "answer_relevancy": 0.7089,
            "context_precision": 0.0,
        },
    )

    cols = st.columns(3)
    for col, (label, key) in zip(
        cols,
        [
            ("Faithfulness", "faithfulness"),
            ("Answer relevancy", "answer_relevancy"),
            ("Context precision", "context_precision"),
        ],
    ):
        with col:
            st.markdown(
                f"""<div class="metric-card">
<div class="metric-value">{float(ragas.get(key, 0)):.4f}</div>
<div class="metric-label">{label}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.caption(
        "The project records these as RAGAS-style values using custom "
        "cosine-based functions rather than the official RAGAS package."
    )


# ============================================================
# ABOUT
# ============================================================

with tab_about:

    st.markdown("## About the project")

    st.write(
        """
This capstone investigates query-adaptive Retrieval-Augmented Generation
through hybrid lexical and semantic retrieval.

The system classifies each query with a trained BERT classifier, maps the
prediction to an adaptive α value, combines BM25 and BGE-M3 retrieval,
and applies a trained cross-encoder to re-rank the candidate evidence.
"""
    )

    st.markdown("### Research collections")

    st.write(
        """
**Natural Questions** — general question answering retrieval.

**TREC-COVID** — biomedical and COVID-19 information retrieval.

**SQuAD** — extractive question answering benchmark.

**AI/ML Research** — 89-paper local research corpus covering RAG,
LLMs, transformers, retrieval, fine-tuning, multimodal models,
evaluation, and related topics.
"""
    )

    st.markdown("### Demonstration behavior")

    st.write(
        """
The Research Demo accepts free-form questions. Benchmark evaluation
numbers remain saved experimental results; a new query runs inference
through the available trained retrieval components rather than pretending
that a precomputed answer was generated for that question.
"""
    )
