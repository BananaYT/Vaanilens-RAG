from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
DEFAULT_INDEX = Path("storage/vaanilens_scam_cases_faiss")
DEFAULT_TEST = Path("data/splits/test.jsonl")
DEFAULT_REPORT = Path("data/reports/retrieval_evaluation.json")
DEFAULT_PREDICTIONS = Path("data/reports/retrieval_predictions.jsonl")
POSITIVE_LABEL = "scam_candidate"
NEGATIVE_LABEL = "benign_candidate"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Test file not found: {path}. Run prepare_scam_dataset.py first.")
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict) and isinstance(row.get("text"), str) and row.get("label") in {POSITIVE_LABEL, NEGATIVE_LABEL}:
                out.append(row)
    if not out:
        raise SystemExit(f"No supported test rows found in {path}.")
    return out


def metric_counts(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    tp = sum(t == POSITIVE_LABEL and p == POSITIVE_LABEL for t, p in zip(y_true, y_pred))
    fp = sum(t == NEGATIVE_LABEL and p == POSITIVE_LABEL for t, p in zip(y_true, y_pred))
    tn = sum(t == NEGATIVE_LABEL and p == NEGATIVE_LABEL for t, p in zip(y_true, y_pred))
    fn = sum(t == POSITIVE_LABEL and p == NEGATIVE_LABEL for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "count": len(y_true),
        "accuracy": round(accuracy, 6),
        "precision_scam_candidate": round(precision, 6),
        "recall_scam_candidate": round(recall, 6),
        "f1_scam_candidate": round(f1, 6),
        "false_positive_rate": round(fpr, 6),
        "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a simple k-nearest-neighbour label baseline against the held-out test split.")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX))
    parser.add_argument("--test-file", default=str(DEFAULT_TEST))
    parser.add_argument("--report-file", default=str(DEFAULT_REPORT))
    parser.add_argument("--predictions-file", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("-k", type=int, default=5, help="Number of nearest training examples to vote over")
    parser.add_argument("--limit", type=int, default=0, help="Maximum test rows to score; 0 means all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()

    if args.k <= 0 or args.batch_size <= 0 or args.limit < 0:
        raise SystemExit("-k and --batch-size must be > 0; --limit must be >= 0.")
    index_dir = Path(args.index_dir)
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Index not found at {manifest_path}. Run ingest_scam.py --target cases first.")
    index_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if index_manifest.get("index_type") != "cases":
        raise SystemExit("The supplied index is not a scam-cases index.")
    if not index_manifest.get("input_file", "").replace("\\", "/").endswith("data/splits/train.jsonl"):
        logging.warning("The index manifest input_file is not the standard train split; ensure your test rows were excluded from index creation.")

    test_rows = read_jsonl(Path(args.test_file))
    if args.limit:
        test_rows = test_rows[:args.limit]
    model_name = index_manifest.get("embedding_model")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": args.device},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    k = min(args.k, int(getattr(db.index, "ntotal", args.k)))
    logging.info("Scoring %d held-out rows against %d training vectors using k=%d", len(test_rows), db.index.ntotal, k)

    predictions: list[dict[str, Any]] = []
    truth: list[str] = []
    predicted: list[str] = []
    language_truth: dict[str, list[str]] = defaultdict(list)
    language_predicted: dict[str, list[str]] = defaultdict(list)

    # Embed batches together for better performance than calling the embedding model once per test row.
    for start in range(0, len(test_rows), args.batch_size):
        batch = test_rows[start:start + args.batch_size]
        vectors = np.asarray(embeddings.embed_documents([r["text"] for r in batch]), dtype=np.float32)
        distances, indices = db.index.search(vectors, k)
        for local_idx, row in enumerate(batch):
            neighbour_labels: list[str] = []
            neighbour_info: list[dict[str, Any]] = []
            for rank, db_idx in enumerate(indices[local_idx]):
                if int(db_idx) < 0:
                    continue
                docstore_id = db.index_to_docstore_id.get(int(db_idx))
                if docstore_id is None:
                    continue
                doc = db.docstore.search(docstore_id)
                if not hasattr(doc, "metadata"):
                    continue
                label = doc.metadata.get("label")
                if label not in {POSITIVE_LABEL, NEGATIVE_LABEL}:
                    continue
                neighbour_labels.append(label)
                neighbour_info.append({
                    "id": doc.metadata.get("id", ""),
                    "label": label,
                    "distance": float(distances[local_idx][rank]),
                })
            counts = Counter(neighbour_labels)
            if not counts:
                pred = NEGATIVE_LABEL
            elif counts[POSITIVE_LABEL] > counts[NEGATIVE_LABEL]:
                pred = POSITIVE_LABEL
            elif counts[NEGATIVE_LABEL] > counts[POSITIVE_LABEL]:
                pred = NEGATIVE_LABEL
            else:
                pred = neighbour_labels[0]  # deterministic tie-break: nearest neighbor

            true = row["label"]
            truth.append(true)
            predicted.append(pred)
            language = str(row.get("language", "unknown"))
            language_truth[language].append(true)
            language_predicted[language].append(pred)
            predictions.append({
                "id": row.get("id", ""),
                "text": row["text"],
                "true_label": true,
                "predicted_label": pred,
                "correct": bool(true == pred),
                "language": language,
                "neighbours": neighbour_info,
                "note": "Baseline prediction against source labels; not a validated real-world fraud result.",
            })
        logging.info("Evaluated %d/%d test rows", min(start + len(batch), len(test_rows)), len(test_rows))

    by_language = {
        language: metric_counts(language_truth[language], language_predicted[language])
        for language in sorted(language_truth)
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "nearest-neighbour majority-vote baseline for source labels",
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "k": k,
        "test_rows_scored": len(test_rows),
        "index_dir": str(index_dir),
        "index_manifest": str(manifest_path),
        "test_file": args.test_file,
        "overall": metric_counts(truth, predicted),
        "by_language_heuristic": by_language,
        "limitations": [
            "The source labels have not been independently audited by this project.",
            "The source dataset card says its English portion partly derives from a general SMS spam corpus; spam is not always fraud.",
            "Language tags are heuristic estimates because the source file has no per-row language field.",
            "This is a retrieval-neighbour baseline, not a trained scam classifier and not evidence of live-call performance.",
            "A split reduces direct row leakage, but near-duplicate messages across sources/campaigns may still exist.",
        ],
    }
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions_path = Path(args.predictions_file)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n=== VaaniLens retrieval baseline (not a production scam detector) ===")
    print(json.dumps(report["overall"], indent=2))
    print("\nBy heuristic language group:")
    print(json.dumps(by_language, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {report_path}")
    print(f"Per-record predictions saved to: {predictions_path}")
    print("Interpret these numbers against the dataset's source labels only; do not claim real-world scam-call accuracy from this test.")


if __name__ == "__main__":
    main()
