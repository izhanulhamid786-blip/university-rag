import argparse
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.chunker import chunk
from ingest.loader import load_all
from ingest.build_db import _metadata, BATCH_SIZE
from rag.model_loading import (
    clear_broken_proxy_env,
    recover_closed_huggingface_session,
    reset_huggingface_session_if_closed,
)
from rag.settings import get_settings


def build_incremental() -> int:
    settings = get_settings()

    print("Loading documents...")
    documents = load_all()
    chunks = chunk(documents)
    if not chunks:
        print("No chunks found.")
        return 1

    print(f"  {len(documents)} docs -> {len(chunks)} target chunks")

    print("Loading ChromaDB...")
    client = chromadb.PersistentClient(path=str(settings.vector_db_dir))
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    existing_data = collection.get(include=[])
    existing_ids = set(existing_data.get("ids", []))
    
    target_ids = {item["id"] for item in chunks}
    
    ids_to_add = target_ids - existing_ids
    ids_to_remove = existing_ids - target_ids
    
    print(f"  Found {len(existing_ids)} existing chunks in DB.")
    print(f"  Need to ADD/UPDATE: {len(ids_to_add)} chunks.")
    print(f"  Need to REMOVE: {len(ids_to_remove)} chunks.")
    
    if ids_to_remove:
        print(f"Removing {len(ids_to_remove)} obsolete chunks...")
        # delete in batches of 1000
        remove_list = list(ids_to_remove)
        for i in range(0, len(remove_list), 1000):
            collection.delete(ids=remove_list[i:i+1000])

    if not ids_to_add:
        print("Database is up to date!")
        return 0

    chunks_to_embed = [c for c in chunks if c["id"] in ids_to_add]
    texts_to_embed = [c["text"] for c in chunks_to_embed]

    print("Loading embedding model...")
    clear_broken_proxy_env()
    for attempt in range(2):
        reset_huggingface_session_if_closed()
        try:
            model = SentenceTransformer(
                settings.embed_model,
                local_files_only=settings.local_files_only,
            )
            break
        except RuntimeError as exc:
            if attempt == 0 and recover_closed_huggingface_session(exc):
                continue
            raise

    print(f"Embedding {len(texts_to_embed)} new/modified chunks...")
    embeddings = model.encode(
        texts_to_embed,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    ).tolist()

    print("Storing in ChromaDB...")
    for start in range(0, len(chunks_to_embed), BATCH_SIZE):
        batch = chunks_to_embed[start : start + BATCH_SIZE]
        batch_slice = slice(start, start + len(batch))
        collection.upsert(
            documents=texts_to_embed[batch_slice],
            embeddings=embeddings[batch_slice],
            ids=[item["id"] for item in batch],
            metadatas=[_metadata(item) for item in batch],
        )
        print(f"  Upserted batch {start // BATCH_SIZE + 1} ({min(start + BATCH_SIZE, len(chunks_to_embed))}/{len(chunks_to_embed)})")

    print(f"Done. Collection now contains {collection.count()} chunks.")
    return 0

if __name__ == "__main__":
    raise SystemExit(build_incremental())
