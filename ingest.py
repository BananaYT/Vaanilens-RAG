from __future__ import annotations
import argparse, json, logging, re
from pathlib import Path
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
DATASET_ID = 'ai4bharat/IndicCorpV2'
CONFIG = 'indiccorp_v2'
MODELS = {
    'mpnet': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'bge-m3': 'BAAI/bge-m3',
}

def get_text(row):
    for key in ('text', 'content', 'sentence', 'document'):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return re.sub(r'\s+', ' ', val).strip()
    for val in row.values():
        if isinstance(val, str) and val.strip():
            return re.sub(r'\s+', ' ', val).strip()
    return ''

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-rows', type=int, default=1000, help='Rows to scan; 0 means no limit')
    p.add_argument('--min-chars', type=int, default=40)
    p.add_argument('--embedding-model', choices=MODELS, default='mpnet')
    p.add_argument('--chunk-size', type=int, default=750)
    p.add_argument('--chunk-overlap', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    p.add_argument('--output-dir', default='storage/indiccorp_hi_faiss')
    p.add_argument('--cache-dir', default='.cache/huggingface')
    a = p.parse_args()
    if a.max_rows < 0 or a.chunk_size <= 0 or not 0 <= a.chunk_overlap < a.chunk_size:
        raise SystemExit('Invalid row/chunk settings.')

    logging.info('Loading %s / %s in streaming mode', DATASET_ID, CONFIG)
    # IMPORTANT: use the named split, not data_dir="data/eng_Latn".
    ds = load_dataset(
    "text",
    data_files={
        "hin_Deva": "hf://datasets/ai4bharat/IndicCorpV2@main/data/hi-*.txt"
    },
    streaming=True,
    cache_dir=a.cache_dir,
    encoding="utf-8",
)
    splits = list(ds.keys()) if hasattr(ds, 'keys') else []
    logging.info('Available splits: %s', splits)
    if 'hin_Deva' not in splits:
        raise SystemExit("Hindi split 'hin_Deva' not found; check current dataset card/splits above.")

    docs = []
    for i, row in enumerate(ds['hin_Deva']):
        if a.max_rows and i >= a.max_rows:
            break
        txt = get_text(row)
        if len(txt) >= a.min_chars:
            docs.append(Document(page_content=txt, metadata={
                'dataset': DATASET_ID, 'split': 'hin_Deva', 'language': 'hi',
                'source_row_index': i
            }))
        if (i + 1) % 500 == 0:
            logging.info('Scanned %d rows; retained %d', i + 1, len(docs))
    if not docs:
        raise SystemExit('No documents retained. Inspect dataset schema or lower --min-chars.')

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=a.chunk_size, chunk_overlap=a.chunk_overlap, add_start_index=True,
        separators=['\n\n', '\n', '।', '. ', '? ', '! ', ' ', '']
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata['chunk_id'] = f'hi_chunk_{i:09d}'

    model_name = MODELS[a.embedding_model]
    logging.info('Embedding %d chunks with %s on %s', len(chunks), model_name, a.device)
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': a.device},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': a.batch_size},
    )
    db = FAISS.from_documents(chunks, embeddings)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    db.save_local(str(out))
    manifest = {
        'dataset_id': DATASET_ID, 'config': CONFIG, 'available_splits': splits,
        'indexed_splits': ['hin_Deva'],
        'note': 'Published IndicCorpV2 config does not list an English split; this index is Hindi-only.',
        'source_documents': len(docs), 'chunks': len(chunks),
        'chunk_size_chars': a.chunk_size, 'chunk_overlap_chars': a.chunk_overlap,
        'embedding_model': model_name, 'embedding_model_key': a.embedding_model,
        'device': a.device,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    logging.info('Saved index and manifest to %s', out)

if __name__ == '__main__':
    main()
