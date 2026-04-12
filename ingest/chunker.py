import hashlib
import re
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


sys.path.insert(0, str(Path(__file__).parent.parent))


URL_RE = re.compile(r"https?://[^\s'\"<>)]+", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?(?:\d[\s-]?){10,13}")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$|.+\s\|\s.+|\t+")
SIZES = {
    "pdf": (1100, 180),
    "docx": (900, 140),
    "html": (850, 140),
    "htm": (850, 140),
    "csv": (700, 90),
    "xlsx": (700, 90),
    "txt": (850, 120),
    "md": (850, 120),
    "json": (650, 80),
}
SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""]


def _splitter(file_type: str) -> RecursiveCharacterTextSplitter:
    chunk_size, chunk_overlap = SIZES.get(file_type, (800, 120))
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )


def _merge_tiny(parts: list[str], min_len: int = 80) -> list[str]:
    merged = []
    for part in parts:
        clean = part.strip()
        if not clean:
            continue
        if len(clean) < min_len and merged:
            merged[-1] = f"{merged[-1]}\n{clean}".strip()
        else:
            merged.append(clean)
    return merged


def _chunk_links(text: str, doc_links: list[dict]) -> list[dict]:
    lower_text = text.lower()
    seen = set()
    found = []
    for link in doc_links:
        url = link["url"]
        anchor = (link.get("anchor_text") or "").lower()
        if url in seen:
            continue
        if url in text or (len(anchor) > 4 and anchor in lower_text):
            found.append(link)
            seen.add(url)

    for url in URL_RE.findall(text):
        if url not in seen:
            found.append(
                {
                    "url": url,
                    "anchor_text": url,
                    "categories": ["general"],
                    "source": "",
                }
            )
            seen.add(url)
    return found


def _table_row_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if TABLE_LINE_RE.search(line))


def _contact_field_count(text: str) -> int:
    return len(EMAIL_RE.findall(text)) + len(PHONE_RE.findall(text))


def _format_chunk_text(doc: dict, part: str) -> str:
    header = []
    if doc.get("title"):
        header.append(f"Title: {doc['title']}")
    if doc.get("category"):
        header.append(f"Category: {doc['category']}")
    if doc.get("source_url"):
        header.append(f"URL: {doc['source_url']}")
    if doc.get("file_type"):
        header.append(f"Type: {doc['file_type']}")

    prefix = "\n".join(header).strip()
    if prefix:
        return f"{prefix}\n\n{part.strip()}"
    return part.strip()


def chunk(documents: list[dict]) -> list[dict]:
    chunks = []
    for doc in documents:
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        parts = _merge_tiny(_splitter(doc.get("file_type", "txt")).split_text(text))
        for index, part in enumerate(parts):
            chunk_links = _chunk_links(part, doc.get("links", []))
            chunk_text = _format_chunk_text(doc, part)
            table_rows = _table_row_count(part)
            contact_fields = _contact_field_count(part)
            chunk_id = hashlib.md5(
                f"{doc['doc_id']}::{index}::{part[:120]}".encode("utf-8", errors="ignore")
            ).hexdigest()
            chunks.append(
                {
                    "text": chunk_text,
                    "raw_text": part,
                    "id": chunk_id,
                    "doc_id": doc["doc_id"],
                    "source": doc["source"],
                    "source_path": doc.get("source_path", doc["source"]),
                    "source_url": doc.get("source_url"),
                    "title": doc.get("title") or Path(doc["source"]).stem,
                    "category": doc.get("category", "general"),
                    "file_type": doc.get("file_type", "txt"),
                    "source_kind": doc.get("source_kind", "crawler"),
                    "chunk_index": index,
                    "chunk_total": len(parts),
                    "has_links": bool(chunk_links),
                    "links": chunk_links,
                    "scraped_at": doc.get("scraped_at"),
                    "ocr": bool(doc.get("ocr", False)),
                    "has_table": bool(table_rows),
                    "table_row_count": table_rows,
                    "contact_field_count": contact_fields,
                }
            )
    return chunks


def find_links_in_chunks(chunks: list[dict], query: str) -> list[dict]:
    words = re.findall(r"\w+", query.lower())
    seen = {}
    for chunk_item in chunks:
        for link in chunk_item["links"]:
            haystack = f"{link['anchor_text']} {link['url']} {' '.join(link['categories'])}".lower()
            score = sum(1 for word in words if word in haystack)
            if score and (link["url"] not in seen or seen[link["url"]]["_score"] < score):
                seen[link["url"]] = {**link, "_score": score}
    return sorted(seen.values(), key=lambda item: (-item["_score"], item["url"]))


if __name__ == "__main__":
    import sys
    from collections import Counter

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ingest.loader import load_all

    docs = load_all()
    chunks = chunk(docs)
    print(f"Total chunks: {len(chunks)}")
    print("By type:", dict(Counter(item["file_type"] for item in chunks)))
    print(f"With links: {sum(1 for item in chunks if item['has_links'])}")
    for link in find_links_in_chunks(chunks, "admission form cuet ug")[:5]:
        print(f"  {link['anchor_text']} -> {link['url']}")
