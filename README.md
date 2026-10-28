# Corrected IndicCorpV2 streaming ingestion

The previous version failed because it passed `data_dir="data/eng_Latn"`. IndicCorpV2's README maps language data to named dataset splits; Hindi is `hin_Deva` and its files are `data/hi-*.txt`. The published split list does not list an English split, so this version indexes Hindi only. For English/Hinglish, use a separate suitable corpus or consented labelled transcripts.

## Windows PowerShell setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run a small smoke test
```powershell
python ingest.py --max-rows 1000 --embedding-model mpnet
```

## Query the index
```powershell
python retrieve.py "आपका खाता बंद हो जाएगा, OTP बताइए" -k 5
```

## Compare BGE-M3
```powershell
python ingest.py --max-rows 1000 --embedding-model bge-m3 --output-dir storage/indiccorp_hi_bge_m3
```

This is a general language corpus, not a verified scam knowledge base. For real scam detection, create a separate source-attributed knowledge base with verified scam patterns and legitimate counterexamples, and evaluate it on human-labelled call transcripts.

Dataset card: https://huggingface.co/datasets/ai4bharat/IndicCorpV2/blob/main/README.md

---

# VaaniLens Phase 1 & 2: scam dataset + separate retrieval indexes

This extension adds a labelled seed dataset pipeline and two FAISS indexes **without changing `ingest.py`, `retrieve.py`, or `storage/indiccorp_hi_faiss`**.

- `prepare_scam_dataset.py`: downloads the dataset's raw `data.json`, validates the schema/labels, normalizes whitespace, removes exact duplicates, excludes text with conflicting labels, records possible phone/URL/email fields for review, produces a deterministic stratified train/test split, and writes audit reports.
- `data/curated/scam_cases.jsonl`: all accepted, deduplicated records with provenance and source label retained.
- `data/splits/train.jsonl` and `data/splits/test.jsonl`: reproducible stratified split. The index uses **train only**; test remains held out.
- `data/knowledge/scam_guidance.jsonl`: separate, source-linked safety guidance paraphrases.
- `ingest_scam.py`: builds a scam-case FAISS index and a separate official-guidance index.
- `retrieve_scam.py`: retrieves labelled candidates or guidance with source information.
- `evaluate_scam.py`: runs a nearest-neighbour majority-vote baseline against the held-out source labels and reports precision, recall, F1, false-positive rate, accuracy, and language-group metrics.
- `ATTRIBUTION.md`: dataset provenance, licence, and caveats.

## Important limitations of the seed dataset

The source card says the dataset has 3,787 records and declares **CC BY 4.0**. Its English portion partly derives from the UCI SMS Spam Collection and the Hinglish portion includes hand-written patterns. General spam is not always fraud, and SMS examples are not the same as recorded scam-call transcripts. The source JSON contains only `text` and binary `label`; VaaniLens cannot truthfully infer per-row language, exact scam type, or detailed fraud indicators from that source file alone.

To avoid silently treating every row as verified fraud, the importer preserves `source_label` and maps labels to `scam_candidate` / `benign_candidate`; each record has `review_status=source_label_unreviewed`. The language field is a rough script/word-list heuristic, not ground truth. Audit the data before treating it as trusted evidence or making product-level claims. See `ATTRIBUTION.md`.

## Windows PowerShell: run Phase 1

Open PowerShell in the project root, the folder containing `ingest.py`.

If you do not already have an environment, create and activate one:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked by PowerShell policy, run this once in that terminal, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Download and prepare the dataset:

```powershell
python prepare_scam_dataset.py
```

The script downloads `data.json` from a pinned revision of the official Hugging Face repository to `data/raw/bolewara_hinglish_scam_text_dataset/data.json` for repeatability. It does not use `datasets.load_dataset()` because the Hugging Face dataset viewer currently reports a parsing issue for this nested JSON file; the file is parsed directly with Python's JSON parser.

Files created:

```text
data/
├── raw/bolewara_hinglish_scam_text_dataset/data.json
├── curated/scam_cases.jsonl
├── splits/train.jsonl
├── splits/test.jsonl
├── knowledge/scam_guidance.jsonl
└── reports/
    ├── dataset_manifest.json
    ├── invalid_rows.jsonl
    ├── duplicate_records.jsonl
    └── label_conflicts.jsonl
```

Read `data/reports/dataset_manifest.json` to see the actual counts from the downloaded revision. The dataset card describes about 3,787 rows; the pipeline's report is authoritative for the local file because invalid rows, duplicates, and label conflicts may change the retained count.

To explicitly re-download the source file:

```powershell
python prepare_scam_dataset.py --refresh
```

To process an already-downloaded source file without internet access:

```powershell
python prepare_scam_dataset.py --no-download
```

Do not edit the generated train/test split manually. If the source or split changes, rerun preparation and rebuild the case index. The default random seed is `42` and test fraction is `20%` per label.

## Windows PowerShell: run Phase 2

Build both new indexes:

```powershell
python ingest_scam.py --target all --embedding-model mpnet
```

The first run may download the multilingual embedding model if it is not already in the Hugging Face cache. You can choose `--device cuda` if your PyTorch/CUDA setup is working; CPU is the safest default.

New indexes are stored separately:

```text
storage/vaanilens_scam_cases_faiss/
storage/vaanilens_scam_knowledge_faiss/
```

The existing `storage/indiccorp_hi_faiss/` remains untouched.

## Test retrieval

Search for similar labelled examples:

```powershell
python retrieve_scam.py "Your bank account will be blocked. Share OTP" --target cases -k 5
```

Search official guidance separately:

```powershell
python retrieve_scam.py "Do not share OTP with a caller" --target guidance -k 3
```

Search only candidates with one source label:

```powershell
python retrieve_scam.py "KYC account blocked OTP" --target cases --label scam_candidate -k 5
```

Hindi query in PowerShell:

```powershell
python retrieve_scam.py "आपका खाता बंद हो जाएगा, OTP बताइए" --target cases -k 5
```

If your terminal has trouble displaying Hindi, you can set UTF-8 for that PowerShell session:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

FAISS distances are distances in embedding space; **lower means closer, not a probability**. The case index holds candidates from the dataset's training split and returns both scam-like and benign examples, so inspect labels and context together.

## Run the retrieval baseline

Fast smoke test on up to 50 held-out test rows:

```powershell
python evaluate_scam.py --limit 50
```

Full evaluation over the held-out split:

```powershell
python evaluate_scam.py
```

The reports are saved to:

- `data/reports/retrieval_evaluation.json`
- `data/reports/retrieval_predictions.jsonl`

The baseline uses the majority label of the `k` nearest training examples. It reports precision, recall, F1, false-positive rate, accuracy, and separate metrics by heuristic language group. Treat it as a **pipeline sanity check only**: the inputs have source labels that have not been independently audited, and the source includes general SMS spam. It does not establish real-world call-detection performance.

## Existing IndicCorpV2 commands still work

The original Hindi index and commands are unchanged:

```powershell
python ingest.py --max-rows 1000 --embedding-model mpnet
python retrieve.py "आपका खाता बंद हो जाएगा, OTP बताइए" -k 5
```

Use `retrieve.py` only for the general IndicCorpV2 index and `retrieve_scam.py` for the new source-attributed scam/guidance indexes.
