import argparse
import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest.chunker import chunk
from ingest.loader import load_all
from rag.settings import get_settings


BATCH_SIZE = 256


def _metadata(chunk_item: dict) -> dict:
    return {
        "chunk_id": chunk_item["id"],
        "doc_id": chunk_item["doc_id"],
        "title": chunk_item["title"],
        "source": chunk_item["source"],
        "source_path": chunk_item["source_path"],
        "source_url": chunk_item.get("source_url") or "",
        "source_kind": chunk_item.get("source_kind", "crawler"),
        "category": chunk_item["category"],
        "file_type": chunk_item["file_type"],
        "chunk_index": str(chunk_item["chunk_index"]),
        "chunk_total": str(chunk_item["chunk_total"]),
        "has_links": str(bool(chunk_item["has_links"])).lower(),
        "links": json.dumps(chunk_item["links"][:15]),
        "scraped_at": chunk_item.get("scraped_at") or "",
        "ocr": str(bool(chunk_item.get("ocr", False))).lower(),
        "has_table": str(bool(chunk_item.get("has_table", False))).lower(),
        "table_row_count": str(int(chunk_item.get("table_row_count", 0))),
        "contact_field_count": str(int(chunk_item.get("contact_field_count", 0))),
    }


def build(reset: bool = False) -> int:
    settings = get_settings()

    print("Loading documents...")
    documents = load_all()
    chunks = chunk(documents)
    if not chunks:
        print("No chunks found. Add crawler output under data/structured or curated files under data/manual.")
        return 1

    print(f"  {len(documents)} docs -> {len(chunks)} chunks")
    print("Embedding...")
    try:
        model = SentenceTransformer(
            settings.embed_model,
            local_files_only=settings.local_files_only,
        )
    except Exception as exc:
        print(
            "Failed to load the embedding model. "
            "If this machine is offline, make sure the model is already cached locally "
            "or set RAG_LOCAL_FILES_ONLY=false for a one-time download."
        )
        print(f"Details: {exc}")
        return 1
    texts = [item["text"] for item in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    ).tolist()

    print("Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=str(settings.vector_db_dir))
    collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if reset:
        existing_ids = collection.get().get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)
            print(f"  Cleared {len(existing_ids)} existing chunks")

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        batch_slice = slice(start, start + len(batch))
        collection.upsert(
            documents=texts[batch_slice],
            embeddings=embeddings[batch_slice],
            ids=[item["id"] for item in batch],
            metadatas=[_metadata(item) for item in batch],
        )
        print(f"  Upserted batch {start // BATCH_SIZE + 1} ({start + len(batch)}/{len(chunks)})")

    print(f"Done. Collection '{settings.collection_name}' now contains {collection.count()} chunks.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or refresh the Chroma vector database.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing chunks before rebuilding the collection.",
    )
    args = parser.parse_args()
    return build(reset=args.reset)


if __name__ == "__main__":
    raise SystemExit(main())
