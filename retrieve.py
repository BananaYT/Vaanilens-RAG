import argparse, json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    p=argparse.ArgumentParser(); p.add_argument('query'); p.add_argument('--index-dir',default='storage/indiccorp_hi_faiss'); p.add_argument('-k',type=int,default=5); p.add_argument('--device',choices=['cpu','cuda','mps'],default='cpu'); a=p.parse_args()
    mf=Path(a.index_dir)/'manifest.json'
    if not mf.exists(): raise SystemExit(f'No manifest at {mf}; run ingest.py first.')
    m=json.loads(mf.read_text(encoding='utf-8'))
    emb=HuggingFaceEmbeddings(model_name=m['embedding_model'],model_kwargs={'device':a.device},encode_kwargs={'normalize_embeddings':True})
    db=FAISS.load_local(a.index_dir,emb,allow_dangerous_deserialization=True)
    for n,(doc,score) in enumerate(db.similarity_search_with_score(a.query,k=a.k),1):
        print(f'\n--- Result {n} | distance={score:.4f} ---\nMetadata: {doc.metadata}\n{doc.page_content}')
if __name__=='__main__': main()
