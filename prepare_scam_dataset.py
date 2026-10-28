from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DATASET_ID = "bolewara/hinglish-scam-text-dataset"
DATASET_URL = "https://huggingface.co/datasets/bolewara/hinglish-scam-text-dataset"
SOURCE_REVISION = "743799229923fc80fd7e9f2c764b5e9bd4ab3c9a"
RAW_URL = f"https://huggingface.co/datasets/bolewara/hinglish-scam-text-dataset/resolve/{SOURCE_REVISION}/data.json"
LICENSE = "CC-BY-4.0"

DEFAULT_RAW = Path("data/raw/bolewara_hinglish_scam_text_dataset/data.json")
DEFAULT_DATA_DIR = Path("data")

# This intentionally stays a small heuristic. The source file has no per-row language field.
HINGLISH_MARKERS = {
    "aap", "aapka", "aapki", "aapke", "aapko", "hai", "hain", "ho", "hoga",
    "hogi", "kar", "karo", "karein", "kijiye", "kripya", "liye", "liye", "mein",
    "mera", "meri", "mere", "tum", "tumhara", "tumhari", "nahi", "nahin", "abhi",
    "paise", "paisa", "bhejo", "bhejiye", "khatam", "band", "khata", "khate", "batao",
    "batayein", "mila", "milega", "jeet", "jeeta", "nikalo", "jama", "sarkari", "bijli",
    "police", "aapke", "turant", "warna", "warna", "kab", "kyun", "kaise", "karna",
    "karo", "raha", "rahi", "rahe", "paye", "pao", "rupaye", "rupay", "haii",
}
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def normalize_text(text: str) -> str:
    """Normalize whitespace for comparison without changing original message meaning."""
    return re.sub(r"\s+", " ", text).strip()


def duplicate_key(text: str) -> str:
    return normalize_text(text).casefold()


def guess_language(text: str) -> str:
    """Heuristic only; the source dataset does not supply a language label per row."""
    if DEVANAGARI_RE.search(text):
        return "hi-Deva"
    words = {w.casefold() for w in WORD_RE.findall(text)}
    if words & HINGLISH_MARKERS:
        return "hi-Latn-or-code-mixed"
    return "en-or-other-Latin"


def normalize_source_label(value: Any) -> int | None:
    # bool is a subclass of int; handle it explicitly for more predictable validation.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, str):
        v = value.strip().casefold()
        mapping = {
            "0": 0, "benign": 0, "legitimate": 0, "ham": 0, "safe": 0,
            "1": 1, "scam": 1, "fraud": 1, "fraudulent": 1, "spam": 1,
        }
        return mapping.get(v)
    return None


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Could not parse {path} as JSON (line {exc.lineno}, column {exc.colno}). "
            "The Hugging Face dataset viewer currently has a parsing issue; make sure the downloaded "
            "data.json file is complete."
        ) from exc
    if isinstance(payload, dict):
        rows = payload.get("rows")
    elif isinstance(payload, list):
        # Support a top-level list if the dataset's file layout changes in the future.
        rows = payload
    else:
        rows = None
    if not isinstance(rows, list):
        raise SystemExit("Unexpected source JSON schema: expected an object with a 'rows' list.")
    return rows


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def download_dataset(path: Path, force: bool = False) -> None:
    if path.exists() and path.stat().st_size > 0 and not force:
        logging.info("Raw source already exists; reusing %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        RAW_URL,
        headers={"User-Agent": "VaaniLens-defensive-research-project/1.0"},
    )
    logging.info("Downloading source dataset from %s", RAW_URL)
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temp_path.open("wb") as out:
            while True:
                chunk = response.read(1024 * 128)
                if not chunk:
                    break
                out.write(chunk)
        if temp_path.stat().st_size < 100:
            raise OSError("Downloaded file is unexpectedly small; refusing to use it.")
        # Validate JSON before replacing a previously downloaded file.
        json.loads(temp_path.read_text(encoding="utf-8-sig"))
        temp_path.replace(path)
    except Exception as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise SystemExit(
            "Dataset download failed. Check your internet connection, then retry. "
            f"Source URL: {RAW_URL}\nDetails: {exc}"
        ) from exc
    logging.info("Downloaded %s bytes to %s", path.stat().st_size, path)


def split_stratified(records: list[dict[str, Any]], test_size: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_label[row["label"]].append(row)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda r: r["id"])
        rng.shuffle(group)
        if len(group) <= 1:
            n_test = 0
        else:
            n_test = max(1, int(round(len(group) * test_size)))
            n_test = min(n_test, len(group) - 1)
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    # Keep stable order in the files, independent of the class group iteration.
    return sorted(train, key=lambda r: r["id"]), sorted(test, key=lambda r: r["id"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download, validate, deduplicate, attribute, and split the VaaniLens scam seed dataset."
    )
    parser.add_argument("--raw-file", default=str(DEFAULT_RAW), help="Path for downloaded source data.json")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Root data directory for curated files/splits/reports")
    parser.add_argument("--test-size", type=float, default=0.20, help="Fraction held out as test data per label (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible split seed")
    parser.add_argument("--refresh", action="store_true", help="Redownload the raw data.json even if it already exists")
    parser.add_argument("--no-download", action="store_true", help="Fail if the raw data file is missing instead of downloading it")
    args = parser.parse_args()

    if not 0.05 <= args.test_size <= 0.4:
        raise SystemExit("--test-size must be between 0.05 and 0.40.")
    raw_path = Path(args.raw_file)
    data_dir = Path(args.data_dir)
    if not args.no_download:
        download_dataset(raw_path, force=args.refresh)
    elif not raw_path.exists():
        raise SystemExit(f"Raw dataset not found: {raw_path}. Remove --no-download to download it.")

    source_rows = read_json_rows(raw_path)
    logging.info("Source JSON rows: %d", len(source_rows))

    valid_records: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    for row_idx, row in enumerate(source_rows):
        if not isinstance(row, dict):
            invalid_records.append({"source_row_index": row_idx, "reason": "row_not_object", "raw_value": repr(row)[:500]})
            continue
        text = row.get("text")
        source_label = normalize_source_label(row.get("label"))
        if not isinstance(text, str) or not normalize_text(text):
            invalid_records.append({"source_row_index": row_idx, "reason": "missing_or_empty_text", "raw_value": row})
            continue
        if source_label is None:
            invalid_records.append({"source_row_index": row_idx, "reason": "unsupported_label", "raw_value": row})
            continue

        clean_text = normalize_text(text)
        label = "scam_candidate" if source_label == 1 else "benign_candidate"
        valid_records.append({
            "id": f"BOL-{row_idx:06d}",
            "text": clean_text,
            "label": label,
            "source_label": source_label,
            "source_label_meaning": "1=source scam/fraud label; 0=source benign label",
            "language": guess_language(clean_text),
            "language_is_heuristic": True,
            "channel": "sms_or_text",
            "scam_type": None,
            "indicators": [],
            "source_type": "external_dataset",
            "source_dataset_id": DATASET_ID,
            "source_dataset_revision": SOURCE_REVISION,
            "source_url": DATASET_URL,
            "source_record_index": row_idx,
            "license": LICENSE,
            "review_status": "source_label_unreviewed",
            "is_synthetic": None,
            "possible_phone_number": bool(PHONE_RE.search(clean_text)),
            "possible_url": bool(URL_RE.search(clean_text)),
            "possible_email": bool(EMAIL_RE.search(clean_text)),
            "label_caveat": (
                "Dataset source label preserved. The source card says its English portion partly derives from a general SMS spam corpus; "
                "spam is not always fraud. Review before treating this example as verified financial-scam evidence."
            ),
        })

    # Group by normalized case-insensitive text, then remove exact duplicates. If same text occurs with
    # conflicting labels, exclude all versions rather than choosing a label arbitrarily.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_records:
        groups[duplicate_key(row["text"])].append(row)
    retained: list[dict[str, Any]] = []
    duplicate_records: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    duplicate_count = 0
    for key, group in groups.items():
        label_set = {r["source_label"] for r in group}
        if len(label_set) > 1:
            conflicts.append({
                "normalized_text_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "reason": "same_normalized_text_has_conflicting_source_labels",
                "records": group,
            })
            continue
        # Source row index order means first occurrence is retained deterministically.
        group.sort(key=lambda r: r["source_record_index"])
        retained.append(group[0])
        for duplicate in group[1:]:
            duplicate_count += 1
            duplicate_records.append({
                "kept_id": group[0]["id"],
                "duplicate_id": duplicate["id"],
                "normalized_text_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "reason": "exact_duplicate_after_whitespace_and_case_normalization",
            })

    retained.sort(key=lambda r: r["source_record_index"])
    train, test = split_stratified(retained, args.test_size, args.seed)

    curated_file = data_dir / "curated" / "scam_cases.jsonl"
    train_file = data_dir / "splits" / "train.jsonl"
    test_file = data_dir / "splits" / "test.jsonl"
    write_jsonl(curated_file, retained)
    write_jsonl(train_file, train)
    write_jsonl(test_file, test)

    report_dir = data_dir / "reports"
    write_jsonl(report_dir / "invalid_rows.jsonl", invalid_records)
    write_jsonl(report_dir / "duplicate_records.jsonl", duplicate_records)
    write_jsonl(report_dir / "label_conflicts.jsonl", conflicts)

    label_counts = Counter(r["label"] for r in retained)
    language_counts = Counter(r["language"] for r in retained)
    by_label_language: dict[str, dict[str, int]] = {}
    for label in sorted(label_counts):
        by_label_language[label] = dict(Counter(r["language"] for r in retained if r["label"] == label))
    possible_pii = sum(
        1 for r in retained if r["possible_phone_number"] or r["possible_email"] or r["possible_url"]
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset_id": DATASET_ID,
        "source_dataset_revision": SOURCE_REVISION,
        "source_dataset_url": DATASET_URL,
        "raw_file": str(raw_path),
        "license": LICENSE,
        "license_note": "CC BY 4.0; retain attribution to Anuj Bolewar and the dataset URL when redistributing derived data.",
        "modifications": "Whitespace normalization, case-insensitive exact-deduplication, exclusion of conflicting-label text groups, conservative source-label mapping, heuristic language grouping, and deterministic stratified train/test split.",
        "raw_rows": len(source_rows),
        "valid_rows_before_deduplication": len(valid_records),
        "retained_unique_rows": len(retained),
        "duplicate_rows_removed": duplicate_count,
        "conflicting_label_text_groups_excluded": len(conflicts),
        "invalid_rows_skipped": len(invalid_records),
        "label_counts": dict(label_counts),
        "language_counts_heuristic": dict(language_counts),
        "label_language_counts": by_label_language,
        "possible_pii_rows_for_review": possible_pii,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_size_requested": args.test_size,
        "split_seed": args.seed,
        "split_method": "deterministic stratified random split by source label after deduplication",
        "language_method": "heuristic based on script and a small Romanized-Hindi word list; not ground truth",
        "label_mapping": {"1": "scam_candidate", "0": "benign_candidate"},
        "label_caveat": (
            "Original source label preserved. The dataset card says the English portion partly derives from the SMS Spam Collection. "
            "Spam may be promotional rather than fraudulent. Source labels are unreviewed and are not verified call-transcript evidence."
        ),
        "retrieval_policy": "Only data/splits/train.jsonl should be indexed; data/splits/test.jsonl remains held out for evaluation.",
    }
    (report_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    logging.info("Prepared %d unique records (removed %d duplicates; excluded %d conflicting text groups).", len(retained), duplicate_count, len(conflicts))
    logging.info("Labels: %s", dict(label_counts))
    logging.info("Heuristic language counts: %s", dict(language_counts))
    logging.info("Train: %d | Test: %d", len(train), len(test))
    logging.info("Curated dataset: %s", curated_file)
    logging.info("Split files: %s, %s", train_file, test_file)
    logging.info("Dataset report: %s", report_dir / "dataset_manifest.json")
    if possible_pii:
        logging.warning("%d retained records contain a possible phone number, URL, or email. Review before publishing/exporting data.", possible_pii)
    if not retained:
        raise SystemExit("No valid source records remained after validation and deduplication.")


if __name__ == "__main__":
    main()
