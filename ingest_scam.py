from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MODELS = {
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge-m3": "BAAI/bge-m3",
}
DEFAULT_CASES_FILE = Path("data/splits/train.jsonl")
DEFAULT_GUIDANCE_FILE = Path("data/knowledge/scam_guidance.jsonl")
DEFAULT_CASES_INDEX = Path("storage/vaanilens_scam_cases_faiss")
DEFAULT_GUIDANCE_INDEX = Path("storage/vaanilens_scam_knowledge_faiss")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Required input file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise SystemExit(f"Invalid document at {path}:{line_no}: each row must have non-empty string 'text'.")
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def to_documents(rows: list[dict[str, Any]], index_type: str) -> list[Document]:
    docs: list[Document] = []
    for row in rows:
        if index_type == "cases":
            metadata = {
                "id": str(row.get("id", "")),
                "label": str(row.get("label", "unknown")),
                "source_label": str(row.get("source_label", "unknown")),
                "language": str(row.get("language", "unknown")),
                "channel": str(row.get("channel", "unknown")),
                "source_type": str(row.get("source_type", "unknown")),
                "source_dataset_id": str(row.get("source_dataset_id", "unknown")),
                "source_url": str(row.get("source_url", "")),
                "source_record_index": int(row.get("source_record_index", -1)),
                "license": str(row.get("license", "unknown")),
                "review_status": str(row.get("review_status", "unknown")),
                "label_caveat": str(row.get("label_caveat", "")),
            }
        else:
            metadata = {
                "id": str(row.get("id", "")),
                "title": str(row.get("title", "")),
                "topic": str(row.get("topic", "unknown")),
                "source_name": str(row.get("source_name", "unknown")),
                "source_url": str(row.get("source_url") or ""),
                "source_date": str(row.get("source_date") or ""),
                "source_type": str(row.get("source_type", "unknown")),
                "license_note": str(row.get("license_note", "")),
                "review_status": str(row.get("review_status", "unknown")),
            }
        docs.append(Document(page_content=row["text"].strip(), metadata=metadata))
    return docs


def build_index(
    *,
    rows: list[dict[str, Any]],
    index_type: str,
    output_dir: Path,
    embedding_model: str,
    device: str,
    batch_size: int,
    input_file: Path,
    embeddings: HuggingFaceEmbeddings,
) -> None:
    docs = to_documents(rows, index_type)
    logging.info("Embedding %d %s records with %s on %s", len(docs), index_type, embedding_model, device)
    db = FAISS.from_documents(docs, embeddings)
    output_dir.mkdir(parents=True, exist_ok=True)
    db.save_local(str(output_dir))

    manifest = {
        "index_type": index_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_indexed": len(docs),
        "embedding_model": embedding_model,
        "embedding_model_key": None,
        "device": device,
        "batch_size": batch_size,
        "normalized_embeddings": True,
        "retrieval_score_note": "FAISS score is L2 distance for normalized embeddings; lower is closer, not a probability or confidence percentage.",
        "data_policy": (
            "Case index contains only the training split; test rows remain held out. Source labels are unreviewed candidate labels."
            if index_type == "cases" else
            "Guidance index contains separate source-attributed guidance and must not be treated as labels for individual calls."
        ),
        "source_dataset_id": "bolewara/hinglish-scam-text-dataset" if index_type == "cases" else None,
        "license": "CC-BY-4.0 for case dataset; see ATTRIBUTION.md" if index_type == "cases" else "Per-record source URLs and notes are stored in data/knowledge/scam_guidance.jsonl",
        "input_file": str(input_file),
    }
    # Keep the model key in the manifest for easier reproducibility.
    for key, value in MODELS.items():
        if value == embedding_model:
            manifest["embedding_model_key"] = key
            break
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Saved %s index and manifest to %s", index_type, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VaaniLens scam-case and official-guidance FAISS indexes (separate from IndicCorpV2).")
    parser.add_argument("--target", choices=["all", "cases", "guidance"], default="all")
    parser.add_argument("--embedding-model", choices=MODELS, default="mpnet")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cases-file", default=str(DEFAULT_CASES_FILE))
    parser.add_argument("--guidance-file", default=str(DEFAULT_GUIDANCE_FILE))
    parser.add_argument("--cases-index-dir", default=str(DEFAULT_CASES_INDEX))
    parser.add_argument("--guidance-index-dir", default=str(DEFAULT_GUIDANCE_INDEX))
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero.")

    cases: list[dict[str, Any]] | None = None
    guidance: list[dict[str, Any]] | None = None
    if args.target in {"all", "cases"}:
        cases = read_jsonl(Path(args.cases_file))
        if not any(row.get("label") in {"scam_candidate", "benign_candidate"} for row in cases):
            raise SystemExit("Case data does not contain expected labels. Run prepare_scam_dataset.py first.")
    if args.target in {"all", "guidance"}:
        guidance = read_jsonl(Path(args.guidance_file))

    model_name = MODELS[args.embedding_model]
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": args.device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": args.batch_size},
    )
    if cases is not None:
        build_index(
            rows=cases,
            index_type="cases",
            output_dir=Path(args.cases_index_dir),
            embedding_model=model_name,
            device=args.device,
            batch_size=args.batch_size,
            input_file=Path(args.cases_file),
            embeddings=embeddings,
        )
    if guidance is not None:
        build_index(
            rows=guidance,
            index_type="guidance",
            output_dir=Path(args.guidance_index_dir),
            embedding_model=model_name,
            device=args.device,
            batch_size=args.batch_size,
            input_file=Path(args.guidance_file),
            embeddings=embeddings,
        )


if __name__ == "__main__":
    main()
