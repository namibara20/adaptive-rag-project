import json
import pickle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "data" / "RAG_Project"

OUTPUT = ROOT / "precomputed_demo.json"


def load_corpus(path):
    return pd.read_csv(
        path,
        usecols=["chunk_id", "doc_id", "chunk_idx", "text", "title"],
    ).fillna("")


def corpus_lookup(df):
    return {
        str(row["chunk_id"]): {
            "chunk_id": str(row["chunk_id"]),
            "doc_id": str(row["doc_id"]),
            "chunk_idx": int(row["chunk_idx"]),
            "title": str(row["title"]),
            "text": str(row["text"]),
        }
        for _, row in df.iterrows()
    }


def attach_results(results, lookup, top_k=5):
    output = []

    for chunk_id, score in results[:top_k]:
        chunk_id = str(chunk_id)

        item = {
            "chunk_id": chunk_id,
            "score": float(score),
        }

        if chunk_id in lookup:
            item.update(lookup[chunk_id])
        else:
            item["title"] = ""
            item["text"] = ""

        output.append(item)

    return output


def build_nq(retrievals):
    query_file = PROJECT / "data" / "nq" / "queries.csv"
    corpus_file = PROJECT / "data" / "nq" / "corpus_chunked.csv"

    questions = pd.read_csv(query_file).fillna("")
    corpus = corpus_lookup(load_corpus(corpus_file))

    # A manageable representative set for the teacher demo.
    # The full 2,924-query retrieval artifact remains untouched.
    selected_ids = [
        "test0",
        "test1",
        "test2",
    ]

    # Add more real questions deterministically.
    all_ids = questions["query_id"].astype(str).tolist()

    for qid in all_ids:
        if qid not in selected_ids:
            selected_ids.append(qid)

        if len(selected_ids) >= 30:
            break

    question_map = {
        str(row["query_id"]): str(row["text"])
        for _, row in questions.iterrows()
    }

    records = []

    for qid in selected_ids:
        if qid not in retrievals["nq_bm25_retrieved"]:
            continue

        records.append(
            {
                "query_id": qid,
                "question": question_map.get(qid, ""),
                "bm25": attach_results(
                    retrievals["nq_bm25_retrieved"][qid],
                    corpus,
                ),
                "dense": attach_results(
                    retrievals["nq_dense_retrieved"][qid],
                    corpus,
                ),
            }
        )

    return records


def build_trec(retrievals):
    query_file = PROJECT / "data" / "trec_covid" / "queries.csv"
    corpus_file = PROJECT / "data" / "trec_covid" / "corpus_chunked.csv"

    questions = pd.read_csv(query_file).fillna("")
    corpus = corpus_lookup(load_corpus(corpus_file))

    question_map = {
        str(row["query_id"]): str(row["text"])
        for _, row in questions.iterrows()
    }

    records = []

    for qid, bm25_results in retrievals["trec_bm25_retrieved"].items():
        qid = str(qid)

        records.append(
            {
                "query_id": qid,
                "question": question_map.get(qid, ""),
                "bm25": attach_results(
                    bm25_results,
                    corpus,
                ),
                "dense": attach_results(
                    retrievals["trec_dense_retrieved"].get(qid, []),
                    corpus,
                ),
            }
        )

    return records


def build_squad():
    qa_file = PROJECT / "data" / "squad" / "qa_pairs.csv"
    corpus_file = PROJECT / "data" / "squad" / "corpus_chunked.csv"

    qa = pd.read_csv(qa_file).fillna("")
    corpus_df = load_corpus(corpus_file)

    # SQuAD QA pairs contain the context document ID.
    corpus_by_doc = {
        str(row["doc_id"]): {
            "chunk_id": str(row["chunk_id"]),
            "doc_id": str(row["doc_id"]),
            "chunk_idx": int(row["chunk_idx"]),
            "title": str(row["title"]),
            "text": str(row["text"]),
        }
        for _, row in corpus_df.iterrows()
    }

    records = []

    # Keep the demo compact while using real SQuAD examples.
    for _, row in qa.head(30).iterrows():
        doc_id = str(row["context_doc_id"])

        answers = row["answers"]

        if isinstance(answers, str):
            try:
                answers = json.loads(answers.replace("'", '"'))
            except Exception:
                answers = [answers]

        if not isinstance(answers, list):
            answers = [str(answers)]

        records.append(
            {
                "question_id": str(row["question_id"]),
                "question": str(row["question"]),
                "reference_answers": [
                    str(a) for a in answers
                ],
                "answer_start": str(row["answer_start"]),
                "context": corpus_by_doc.get(
                    doc_id,
                    {
                        "chunk_id": "",
                        "doc_id": doc_id,
                        "chunk_idx": 0,
                        "title": "",
                        "text": "",
                    },
                ),
            }
        )

    return records


def build_aiml():
    path = ROOT / "data" / "chunks_aiml.csv"

    df = pd.read_csv(path).fillna("")

    records = []

    for _, row in df.iterrows():
        records.append(
            {
                "chunk_id": str(row["chunk_id"]),
                "doc_id": str(row["doc_id"]),
                "chunk_idx": int(row["chunk_idx"]),
                "title": str(row["title"]),
                "authors": str(row["authors"]),
                "paper_id": str(row["paper_id"]),
                "text": str(row["text"]),
            }
        )

    return records


def main():
    retrieval_file = PROJECT / "retrieval_runs.pkl"

    print("Loading saved retrieval artifacts...")
    with open(retrieval_file, "rb") as f:
        retrievals = pickle.load(f)

    print("Building NQ...")
    nq = build_nq(retrievals)

    print("Building TREC-COVID...")
    trec = build_trec(retrievals)

    print("Building SQuAD...")
    squad = build_squad()

    print("Building AI/ML corpus...")
    aiml = build_aiml()

    demo = {
        "metadata": {
            "type": "precomputed_research_demo",
            "description": (
                "Precomputed results generated from the Adaptive RAG "
                "project's saved retrieval artifacts and benchmark data. "
                "This file does not run the retrieval models live."
            ),
            "source": "Adaptive RAG capstone project",
        },
        "datasets": {
            "nq": nq,
            "trec_covid": trec,
            "squad": squad,
            "aiml": aiml,
        },
    }

    print("Writing:", OUTPUT)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(
            demo,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("DONE")
    print("NQ demo questions:", len(nq))
    print("TREC-COVID questions:", len(trec))
    print("SQuAD questions:", len(squad))
    print("AI/ML papers:", len(aiml))
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()