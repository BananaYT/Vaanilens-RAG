from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

MODELS = {
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge-m3": "BAAI/bge-m3",
}
DEFAULT_INDEXES = {
    "cases": Path("storage/vaanilens_scam_cases_faiss"),
    "guidance": Path("storage/vaanilens_scam_knowledge_faiss"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve examples from the separate VaaniLens scam index or official-guidance index.")
    parser.add_argument("query", help="Text to search for. Use quotes in PowerShell for a multi-word query.")
    parser.add_argument("--target", choices=["cases", "guidance"], default="cases")
    parser.add_argument("--index-dir", default=None, help="Optional custom index directory")
    parser.add_argument("-k", type=int, default=5, help="Number of results to print")
    parser.add_argument("--label", choices=["scam_candidate", "benign_candidate"], default=None, help="Optional label filter (cases index only)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()

    if args.k <= 0:
        raise SystemExit("-k must be greater than zero.")
    if args.target == "guidance" and args.label:
        raise SystemExit("--label applies only to --target cases.")

    index_dir = Path(args.index_dir) if args.index_dir else DEFAULT_INDEXES[args.target]
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"No index manifest at {manifest_path}. Build it first with ingest_scam.py.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_name = manifest.get("embedding_model")
    if not model_name:
        raise SystemExit(f"Index manifest {manifest_path} has no embedding_model.")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": args.device},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)

    # For label-filtered searches, request the full ranked index so a less common class
    # is not accidentally missed just because the first 50 neighbours were the other label.
    search_k = int(getattr(db.index, "ntotal", args.k)) if args.label else args.k
    results = db.similarity_search_with_score(args.query, k=max(1, search_k))
    if args.label:
        results = [(doc, score) for doc, score in results if doc.metadata.get("label") == args.label]
    results = results[:args.k]
    if not results:
        print("No matching documents found. Try removing --label, increasing -k, or checking the index contents.")
        return

    print(f"Index type: {manifest.get('index_type', args.target)} | source records indexed: {manifest.get('records_indexed', 'unknown')}")
    print("Distance note: lower L2 distance means closer in embedding space; it is not a scam probability.\n")
    for n, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        if args.target == "cases":
            print(f"--- Result {n} | distance={float(score):.4f} | label={meta.get('label', 'unknown')} ---")
            print(f"ID: {meta.get('id', '')} | language heuristic: {meta.get('language', 'unknown')} | review: {meta.get('review_status', 'unknown')}")
            print(f"Source: {meta.get('source_dataset_id', '')} | record index: {meta.get('source_record_index', '')}")
            if meta.get("source_url"):
                print(f"Source URL: {meta['source_url']}")
            if meta.get("label_caveat"):
                print(f"Label caveat: {meta['label_caveat']}")
        else:
            print(f"--- Result {n} | distance={float(score):.4f} | topic={meta.get('topic', 'unknown')} ---")
            print(f"ID: {meta.get('id', '')} | title: {meta.get('title', '')} | source: {meta.get('source_name', '')}")
            if meta.get("source_url"):
                print(f"Source URL: {meta['source_url']}")
            if meta.get("review_status"):
                print(f"Review status: {meta['review_status']}")
        print(doc.page_content)
        print()

    if args.target == "cases":
        print("CAUTION: source labels are unreviewed candidate labels; this dataset includes general SMS spam as well as scam-like examples.")
        print("Use retrieved rows as contextual examples, not proof that the current message or call is fraudulent.")


if __name__ == "__main__":
    main()
