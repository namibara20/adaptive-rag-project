import os
import json
import gc
import numpy as np
import pandas as pd
import torch
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
)


# ============================================================
# PATHS / CONFIG
# ============================================================

DATA_ROOT = os.environ.get(
    "RAG_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "data", "RAG_Project")
)

OUTPUT_DIR = os.environ.get(
    "RAG_OUTPUT_DIR",
    os.path.join(os.path.dirname(__file__), "outputs")
)

DENSE_MODEL = "BAAI/bge-m3"
CLASSIFIER_MODEL = "bert-base-uncased"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_RETRIEVAL = 20
TOP_RERANKED = 5
DENSE_TOP = 100
BM25_TOP = 100

MAX_SEQ_CLF = 128
MAX_SEQ_RR = 256

ALPHA_MAP = {
    0: 0.2,
    1: 0.5,
    2: 0.8,
}

LABEL_MAP = {
    0: "keyword",
    1: "mixed",
    2: "semantic",
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[RAG BACKEND] Device: {DEVICE}")
print(f"[RAG BACKEND] Output directory: {OUTPUT_DIR}")


# ============================================================
# DATASET PATHS
# ============================================================

DATASETS = {
    "trec": {
        "name": "TREC-COVID",
        "path": os.path.join(DATA_ROOT, "data", "trec_covid"),
        "embedding": os.path.join(OUTPUT_DIR, "embeddings_trec.npy"),
    },
    "nq": {
        "name": "Natural Questions",
        "path": os.path.join(DATA_ROOT, "data", "nq"),
        "embedding": os.path.join(OUTPUT_DIR, "embeddings_nq.npy"),
    },
    "squad": {
        "name": "SQuAD",
        "path": os.path.join(DATA_ROOT, "data", "squad"),
        "embedding": os.path.join(OUTPUT_DIR, "embeddings_squad.npy"),
    },
}


# ============================================================
# HELPERS
# ============================================================

def load_chunks(dataset_key):
    cfg = DATASETS[dataset_key]

    path = os.path.join(cfg["path"], "corpus_chunked.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Chunk file not found: {path}"
        )

    df = pd.read_csv(path)

    required = {"chunk_id", "doc_id", "text"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    df = df[["chunk_id", "doc_id", "text"]].copy()

    df["chunk_id"] = df["chunk_id"].astype(str)
    df["doc_id"] = df["doc_id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)

    # Add paper titles from the original corpus metadata.
    # corpus_chunked.csv contains the retrieval chunks, while
    # corpus.csv contains the document-level title.
    corpus_path = os.path.join(cfg["path"], "corpus_metadata.csv")

    if os.path.exists(corpus_path):
        meta = pd.read_csv(
            corpus_path,
            usecols=["doc_id", "title"]
        )

        meta["doc_id"] = meta["doc_id"].astype(str)
        meta["title"] = meta["title"].fillna("").astype(str)

        meta = meta.drop_duplicates(
            subset=["doc_id"]
        )

        df = df.merge(
            meta,
            on="doc_id",
            how="left"
        )
    else:
        df["title"] = ""

    df["title"] = df["title"].fillna("").astype(str)

    return df


def tokenize_bm25(text):
    return str(text).lower().split()


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)

    if len(scores) == 0:
        return scores

    mn = float(scores.min())
    mx = float(scores.max())

    if mx - mn < 1e-12:
        return np.zeros_like(scores)

    return (scores - mn) / (mx - mn)


# ============================================================
# Importing this module is cheap; all data/model loading is explicit.
CLF_PATH = os.path.join(OUTPUT_DIR, "clf_model")
RR_PATH = os.path.join(OUTPUT_DIR, "rr_model")
AIML_CORPUS_PATH = os.environ.get(
    "RAG_AIML_CORPUS_PATH",
    os.path.join(os.path.dirname(__file__), "data", "chunks_aiml.csv"),
)
DATASETS["aiml"] = {
    "name": "AI/ML Research Papers",
    "path": os.path.dirname(AIML_CORPUS_PATH),
}
CHUNKS = {}
BM25_INDEX = {}
FAISS_INDEX = {}
clf_tokenizer = None
clf_model = None
rr_tokenizer = None
rr_model = None
dense_model = None
_CORE_LOADED = False

def load_core_resources():
    global clf_tokenizer, clf_model, rr_tokenizer, rr_model, dense_model, _CORE_LOADED
    if _CORE_LOADED:
        return
    for path, label in ((CLF_PATH, "classifier"), (RR_PATH, "reranker")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Trained {label} model not found: {path}")
    if not os.path.exists(AIML_CORPUS_PATH):
        raise FileNotFoundError(f"AI/ML research corpus not found: {AIML_CORPUS_PATH}")

    clf_tokenizer = BertTokenizer.from_pretrained(CLF_PATH)
    clf_model = BertForSequenceClassification.from_pretrained(CLF_PATH).to(DEVICE).eval()
    rr_tokenizer = AutoTokenizer.from_pretrained(RR_PATH)
    rr_model = AutoModelForSequenceClassification.from_pretrained(
        RR_PATH, num_labels=1, ignore_mismatched_sizes=True
    ).to(DEVICE).eval()
    dense_model = SentenceTransformer(DENSE_MODEL, device=DEVICE)

    df = pd.read_csv(AIML_CORPUS_PATH)
    required = {"chunk_id", "doc_id", "chunk_idx", "text", "title", "authors", "paper_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{AIML_CORPUS_PATH} is missing columns: {sorted(missing)}")
    for col in required:
        df[col] = df[col].fillna("").astype(str)
    CHUNKS["aiml"] = df.reset_index(drop=True)
    BM25_INDEX["aiml"] = BM25Okapi([tokenize_bm25(x) for x in df["text"]])
    vectors = dense_model.encode(
        df["text"].tolist(), normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    FAISS_INDEX["aiml"] = index
    _CORE_LOADED = True
    print(f"[RAG BACKEND] AI/ML chunks={len(df):,}; papers={df['paper_id'].nunique():,}")

def load_benchmark_resources(dataset_key):
    global dense_model
    if dataset_key not in DATASETS or dataset_key == "aiml":
        raise ValueError(f"Unknown benchmark dataset: {dataset_key}")
    if dataset_key in CHUNKS:
        return
    if dense_model is None:
        dense_model = SentenceTransformer(DENSE_MODEL, device=DEVICE)
    cfg = DATASETS[dataset_key]
    df = load_chunks(dataset_key)
    emb_path = cfg["embedding"]
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
    vectors = np.asarray(np.load(emb_path), dtype=np.float32)
    if len(vectors) != len(df):
        raise ValueError(
            f"Embedding count mismatch for {dataset_key}: "
            f"{len(vectors)} embeddings vs {len(df)} chunks"
        )
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    CHUNKS[dataset_key] = df
    BM25_INDEX[dataset_key] = BM25Okapi([tokenize_bm25(x) for x in df["text"]])
    FAISS_INDEX[dataset_key] = index

# QUERY CLASSIFICATION
# ============================================================

def predict_alpha(query):
    """
    Use the trained BERT classifier from NB03.

    Class mapping is identical to the original pipeline:
      0 -> alpha 0.2 -> keyword
      1 -> alpha 0.5 -> mixed
      2 -> alpha 0.8 -> semantic
    """

    enc = clf_tokenizer(
        str(query),
        max_length=MAX_SEQ_CLF,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    enc = {
        k: v.to(DEVICE)
        for k, v in enc.items()
    }

    with torch.no_grad():
        logits = clf_model(**enc).logits

    pred = torch.argmax(
        logits,
        dim=1
    ).item()

    alpha = ALPHA_MAP[pred]
    label = LABEL_MAP[pred]

    return alpha, label


# ============================================================
# BM25 RETRIEVAL
# ============================================================

def retrieve_bm25(
    query,
    dataset_key,
    top_k=BM25_TOP,
):
    bm25 = BM25_INDEX[dataset_key]
    chunks = CHUNKS[dataset_key]

    scores = bm25.get_scores(
        tokenize_bm25(query)
    )

    top_k = min(top_k, len(scores))

    indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for idx in indices:
        row = chunks.iloc[int(idx)]

        results.append({
            "chunk_id": str(row["chunk_id"]),
            "doc_id": str(row["doc_id"]),
            "text": str(row["text"]),
            "title": str(row.get("title", "")),
            "authors": str(row.get("authors", "")),
            "paper_id": str(row.get("paper_id", "")),
            "score": float(scores[idx]),
        })

    return results


# ============================================================
# DENSE RETRIEVAL
# ============================================================

def retrieve_dense(
    query,
    dataset_key,
    top_k=DENSE_TOP,
):
    index = FAISS_INDEX[dataset_key]
    chunks = CHUNKS[dataset_key]

    query_embedding = dense_model.encode(
        [str(query)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    top_k = min(top_k, len(chunks))

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0],
    ):
        if idx < 0:
            continue

        row = chunks.iloc[int(idx)]

        results.append({
            "chunk_id": str(row["chunk_id"]),
            "doc_id": str(row["doc_id"]),
            "text": str(row["text"]),
            "title": str(row.get("title", "")),
            "authors": str(row.get("authors", "")),
            "paper_id": str(row.get("paper_id", "")),
            "score": float(score),
        })

    return results


# ============================================================
# ADAPTIVE HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieve(
    query,
    dataset_key,
    alpha,
    top_k=TOP_RETRIEVAL,
):
    """
    Reproduce the original pipeline's adaptive hybrid idea:

        hybrid = alpha * dense + (1-alpha) * BM25

    BM25 and dense scores are independently min-max normalized
    before fusion.
    """

    bm25_results = retrieve_bm25(
        query,
        dataset_key,
        top_k=BM25_TOP,
    )

    dense_results = retrieve_dense(
        query,
        dataset_key,
        top_k=DENSE_TOP,
    )

    bm25_scores = {
        r["chunk_id"]: r["score"]
        for r in bm25_results
    }

    dense_scores = {
        r["chunk_id"]: r["score"]
        for r in dense_results
    }

    bm25_ids = list(bm25_scores.keys())
    dense_ids = list(dense_scores.keys())

    bm25_norm_values = normalize_scores(
        list(bm25_scores.values())
    )

    dense_norm_values = normalize_scores(
        list(dense_scores.values())
    )

    bm25_norm = dict(
        zip(
            bm25_ids,
            bm25_norm_values
        )
    )

    dense_norm = dict(
        zip(
            dense_ids,
            dense_norm_values
        )
    )

    candidate_ids = set(
        bm25_ids
    ).union(
        dense_ids
    )

    chunk_lookup = {
        str(row["chunk_id"]): row
        for _, row in CHUNKS[dataset_key].iterrows()
    }

    fused = []

    for chunk_id in candidate_ids:

        bm = bm25_norm.get(
            chunk_id,
            0.0
        )

        de = dense_norm.get(
            chunk_id,
            0.0
        )

        score = (
            alpha * de
            + (1.0 - alpha) * bm
        )

        row = chunk_lookup.get(chunk_id)

        if row is None:
            continue

        fused.append({
            "chunk_id": chunk_id,
            "doc_id": str(row["doc_id"]),
            "text": str(row["text"]),
            "title": str(row.get("title", "")),
            "authors": str(row.get("authors", "")),
            "paper_id": str(row.get("paper_id", "")),
            "score": float(score),
            "bm25_score": float(
                bm25_scores.get(
                    chunk_id,
                    0.0
                )
            ),
            "dense_score": float(
                dense_scores.get(
                    chunk_id,
                    0.0
                )
            ),
        })

    fused.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return fused[:top_k]


# ============================================================
# CROSS-ENCODER RERANKING
# ============================================================

def rerank(
    query,
    candidates,
    top_k=TOP_RERANKED,
    batch_size=32,
):
    """
    Reproduce the original pipeline's cross-encoder reranking.
    """

    if not candidates:
        return []

    scores = []

    rr_model.eval()

    for start in range(
        0,
        len(candidates),
        batch_size,
    ):
        batch = candidates[
            start:start + batch_size
        ]

        queries = [
            str(query)
            for _ in batch
        ]

        passages = [
            str(item["text"])
            for item in batch
        ]

        enc = rr_tokenizer(
            queries,
            passages,
            max_length=MAX_SEQ_RR,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        enc = {
            k: v.to(DEVICE)
            for k, v in enc.items()
        }

        with torch.no_grad():
            logits = rr_model(
                **enc
            ).logits

        batch_scores = (
            logits[:, 0]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        scores.extend(batch_scores)

    ranked = []

    for item, score in zip(
        candidates,
        scores
    ):
        result = dict(item)
        result["rerank_score"] = float(score)
        ranked.append(result)

    ranked.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked[:top_k]


# ============================================================
# PUBLIC QUERY FUNCTION
# ============================================================


# ============================================================
# OPTIONAL ACADEMIC WEB RETRIEVAL
# ============================================================


def resolve_benchmark_source(dataset, title):
    """Resolve an original/public source URL for benchmark evidence."""
    import re
    import requests
    from urllib.parse import quote

    title = str(title or "").strip()
    if not title:
        return {
            "source": "",
            "source_url": "",
        }

    try:
        if dataset in ("nq", "squad"):
            # NQ and SQuAD source passages are Wikipedia-derived.
            url = "https://en.wikipedia.org/wiki/" + quote(
                title.replace(" ", "_"),
                safe="_()"
            )

            response = requests.get(
                url,
                timeout=8,
                allow_redirects=True,
                headers={"User-Agent": "AdaptiveRAGResearch/1.0"},
            )

            if response.status_code == 200:
                return {
                    "source": "Wikipedia",
                    "source_url": response.url,
                }

        elif dataset == "trec":
            # TREC-COVID contains biomedical/scientific documents.
            # Europe PMC can resolve exact paper titles to article records.
            api = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            params = {
                "query": f'TITLE:"{title}"',
                "format": "json",
                "pageSize": 3,
            }

            response = requests.get(
                api,
                params=params,
                timeout=8,
                headers={"User-Agent": "AdaptiveRAGResearch/1.0"},
            )

            if response.ok:
                data = response.json()
                results = data.get("resultList", {}).get("result", [])

                normalized_target = re.sub(
                    r"[^a-z0-9]+", " ",
                    title.lower()
                ).strip()

                for item in results:
                    candidate = str(item.get("title", "")).strip()

                    normalized_candidate = re.sub(
                        r"[^a-z0-9]+", " ",
                        candidate.lower()
                    ).strip()

                    if normalized_candidate == normalized_target:
                        pmcid = item.get("pmcid")
                        pmid = item.get("pmid")
                        doi = item.get("doi")

                        if pmcid:
                            url = (
                                "https://europepmc.org/articles/"
                                + str(pmcid)
                            )
                        elif pmid:
                            url = (
                                "https://europepmc.org/article/MED/"
                                + str(pmid)
                            )
                        elif doi:
                            url = "https://doi.org/" + str(doi)
                        else:
                            url = ""

                        if url:
                            return {
                                "source": "Europe PMC",
                                "source_url": url,
                            }

    except Exception:
        pass

    return {
        "source": "",
        "source_url": "",
    }


def search_semantic_scholar(query_text, limit=5):
    """
    Optional Semantic Scholar retrieval.

    This is supplementary to the local AI/ML corpus.
    Network failures and API rate limits are handled gracefully.
    """

    try:
        import requests

        url = "https://api.semanticscholar.org/graph/v1/paper/search"

        params = {
            "query": str(query_text),
            "limit": int(limit),
            "fields": (
                "title,authors,year,url,abstract,openAccessPdf"
            ),
        }

        response = requests.get(
            url,
            params=params,
            timeout=15,
        )

        if response.status_code == 429:
            return []

        if response.status_code != 200:
            return []

        payload = response.json()

        papers = []

        for paper in payload.get("data", []):
            open_access = paper.get(
                "openAccessPdf"
            ) or {}

            papers.append({
                "source": "Semantic Scholar",
                "title": paper.get(
                    "title",
                    ""
                ),
                "authors": ", ".join(
                    author.get("name", "")
                    for author in paper.get(
                        "authors",
                        []
                    )
                    if author.get("name")
                ),
                "year": paper.get(
                    "year"
                ),
                "url": paper.get(
                    "url",
                    ""
                ),
                "abstract": paper.get(
                    "abstract",
                    ""
                ),
                "open_access_pdf": open_access.get(
                    "url",
                    ""
                ),
            })

        return papers

    except Exception:
        return []



# ============================================================
# VERIFIED ARXIV SOURCE RESOLUTION
# ============================================================

def search_arxiv_paper(title, limit=5):
    """
    Resolve a paper title to a verified arXiv record.

    Returns an empty list when the API is unavailable or when
    no sufficiently strong title match is found.
    """

    try:
        import re
        import requests
        import xml.etree.ElementTree as ET

        title = str(title).strip()

        if not title:
            return []

        # Use distinctive title terms rather than sending the
        # entire title as one quoted arXiv expression.
        words = re.findall(
            r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*",
            title
        )

        if len(words) < 3:
            return []

        # Use distinctive title terms and skip common
        # stopwords such as "for", "and", "the", etc.
        stopwords = {
            "a", "an", "and", "are", "as", "at", "by",
            "for", "from", "in", "of", "on", "or",
            "the", "to", "with",
        }

        selected = [
            word
            for word in words
            if word.lower() not in stopwords
        ][:3]

        search_query = " AND ".join(
            f"ti:{word}"
            for word in selected
        )

        response = requests.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": search_query,
                "max_results": int(limit),
                "sortBy": "relevance",
            },
            timeout=20,
        )

        if response.status_code != 200:
            return []

        root = ET.fromstring(
            response.text
        )

        namespace = {
            "atom": "http://www.w3.org/2005/Atom"
        }

        def normalize(value):
            return " ".join(
                str(value).lower().split()
            )

        target = normalize(title)

        matches = []

        for entry in root.findall(
            "atom:entry",
            namespace
        ):
            entry_title = entry.findtext(
                "atom:title",
                default="",
                namespaces=namespace,
            )

            entry_title = " ".join(
                entry_title.split()
            )

            if not entry_title:
                continue

            # Strong exact normalized-title match.
            if normalize(entry_title) != target:
                continue

            entry_id = entry.findtext(
                "atom:id",
                default="",
                namespaces=namespace,
            )

            if not entry_id:
                continue

            arxiv_url = (
                entry_id
                .replace(
                    "http://arxiv.org/",
                    "https://arxiv.org/",
                )
            )

            pdf_url = ""

            for link in entry.findall(
                "atom:link",
                namespace
            ):
                if (
                    link.attrib.get("title")
                    == "pdf"
                ):
                    pdf_url = link.attrib.get(
                        "href",
                        ""
                    )
                    break

            authors = []

            for author in entry.findall(
                "atom:author",
                namespace
            ):
                name = author.findtext(
                    "atom:name",
                    default="",
                    namespaces=namespace,
                )

                if name:
                    authors.append(name)

            published = entry.findtext(
                "atom:published",
                default="",
                namespaces=namespace,
            )

            matches.append({
                "source": "arXiv",
                "title": entry_title,
                "authors": ", ".join(authors),
                "year": (
                    published[:4]
                    if published
                    else None
                ),
                "url": arxiv_url,
                "pdf_url": pdf_url,
            })

        return matches

    except Exception:
        return []



def search_arxiv_external(query_text, max_results=5):
    """
    Search arXiv for external research evidence.

    This is separate from the fixed 89-paper AI/ML corpus.
    It does not modify FAISS/BM25 indexes or benchmark datasets.
    """
    import requests
    import xml.etree.ElementTree as ET
    from urllib.parse import quote_plus

    query_text = str(query_text).strip()

    if not query_text:
        return []

    # Use a compact set of distinctive query terms.
    stopwords = {
        "what", "what's", "how", "why", "when", "where",
        "does", "do", "is", "are", "can", "could", "would",
        "the", "a", "an", "and", "or", "of", "to", "for",
        "in", "on", "with", "from", "about", "explain",
        "please", "me", "tell", "describe"
    }

    terms = []
    for token in query_text.lower().replace("?", " ").split():
        token = token.strip(".,:;!?()[]{}\"'")
        if len(token) >= 3 and token not in stopwords:
            if token not in terms:
                terms.append(token)

    if not terms:
        return []

    # arXiv interprets ungrouped terms as OR. Use an AND query
    # so the external results stay reasonably focused.
    terms = terms[:5]
    search_query = " AND ".join(
        f"all:{quote_plus(term)}"
        for term in terms
    )

    url = "https://export.arxiv.org/api/query"

    try:
        response = requests.get(
            url,
            params={
                "search_query": search_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=20,
        )

        response.raise_for_status()

        root = ET.fromstring(response.text)

        namespace = {
            "atom": "http://www.w3.org/2005/Atom"
        }

        matches = []

        for entry in root.findall(
            "atom:entry",
            namespace,
        ):
            title = entry.findtext(
                "atom:title",
                default="",
                namespaces=namespace,
            ).strip()

            abstract = entry.findtext(
                "atom:summary",
                default="",
                namespaces=namespace,
            ).strip()

            paper_id = entry.findtext(
                "atom:id",
                default="",
                namespaces=namespace,
            ).strip()

            authors = [
                author.findtext(
                    "atom:name",
                    default="",
                    namespaces=namespace,
                ).strip()
                for author in entry.findall(
                    "atom:author",
                    namespace,
                )
            ]

            published = entry.findtext(
                "atom:published",
                default="",
                namespaces=namespace,
            ).strip()

            if not title or not paper_id:
                continue

            arxiv_url = paper_id.replace(
                "http://arxiv.org/",
                "https://arxiv.org/",
            )

            pdf_url = arxiv_url.replace(
                "/abs/",
                "/pdf/",
            )

            matches.append({
                "source": "arXiv",
                "title": " ".join(title.split()),
                "authors": ", ".join(
                    author for author in authors if author
                ),
                "year": (
                    published[:4]
                    if published
                    else None
                ),
                "url": arxiv_url,
                "pdf_url": pdf_url,
                "abstract": " ".join(
                    abstract.split()
                ),
            })

        return matches

    except Exception as exc:
        print(
            f"[RAG BACKEND] arXiv external search failed: {exc}"
        )
        return []


def query(
    query_text,
    dataset="trec",
    top_k=5,
    include_external=False,
):
    """
    Main frontend API.

    Returns:
      query
      dataset
      alpha
      query_type
      retrieved_count
      candidates
      results
    """

    load_core_resources()
    if dataset != "aiml" and dataset not in CHUNKS:
        load_benchmark_resources(dataset)

    if dataset not in CHUNKS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Choose from: {list(CHUNKS.keys())}"
        )

    query_text = str(query_text).strip()

    if not query_text:
        raise ValueError(
            "Query cannot be empty."
        )

    # --------------------------------------------------------
    # 1. BERT query classification
    # --------------------------------------------------------

    alpha, query_type = predict_alpha(
        query_text
    )

    # --------------------------------------------------------
    # 2. Adaptive hybrid retrieval
    # --------------------------------------------------------

    candidates = hybrid_retrieve(
        query_text,
        dataset,
        alpha,
        top_k=TOP_RETRIEVAL,
    )

    # --------------------------------------------------------
    # 3. Cross-encoder reranking
    # --------------------------------------------------------

    results = rerank(
        query_text,
        candidates,
        top_k=top_k,
    )

    # --------------------------------------------------------
    # 4. Optional verified source enrichment for AI/ML papers
    # --------------------------------------------------------

    if dataset in ("trec", "nq", "squad"):
        source_cache = {}

        for item in results:
            title = str(item.get("title", "")).strip()

            if not title:
                continue

            if title not in source_cache:
                source_cache[title] = resolve_benchmark_source(
                    dataset,
                    title,
                )

            source = source_cache[title]

            item["source"] = source.get("source", "")
            item["source_url"] = source.get("source_url", "")

    if dataset == "aiml" and include_external:
        source_cache = {}

        for item in results:
            title = str(
                item.get("title", "")
            ).strip()

            if not title:
                continue

            if title not in source_cache:
                matches = search_arxiv_paper(
                    title
                )
                source_cache[title] = (
                    matches[0]
                    if matches
                    else None
                )

            source = source_cache[title]

            if source:
                item["source"] = source["source"]
                item["source_url"] = source["url"]
                item["pdf_url"] = source["pdf_url"]
            else:
                item["source"] = ""
                item["source_url"] = ""
                item["pdf_url"] = ""

    return {
        "query": query_text,
        "dataset": dataset,
        "dataset_name": DATASETS[dataset]["name"],
        "alpha": float(alpha),
        "query_type": query_type,
        "retrieved_count": len(candidates),
        "candidates": candidates,
        "results": results,
    }


# ============================================================
# EVALUATION RESULTS
# ============================================================

def load_evaluation_results():
    """
    Load the already-completed evaluation results.
    This does NOT rerun evaluation.
    """

    path = os.path.join(
        OUTPUT_DIR,
        "all_results.json"
    )

    if not os.path.exists(path):
        return {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


EVALUATION_RESULTS = load_evaluation_results()


# ============================================================
# SIMPLE HEALTH CHECK
# ============================================================

def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "datasets": {
            key: len(CHUNKS[key])
            for key in CHUNKS
        },
        "models": {
            "dense": DENSE_MODEL,
            "classifier": CLF_PATH,
            "reranker": RR_PATH,
        },
        "evaluation_loaded": bool(
            EVALUATION_RESULTS
        ),
    }
