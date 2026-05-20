import hashlib
import re
import sys
from pathlib import Path
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


sys.path.insert(0, str(Path(__file__).parent.parent))


URL_RE = re.compile(r"https?://[^\s'\"<>)]+", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?:\+91[\s-]?)?(?:\d[\s-]?){10,13}")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$|.+\s\|\s.+|\t+")
PAGE_BREAK_RE = re.compile(r"\n\s*--- PAGE BREAK ---\s*\n", re.I)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,8}(?:[-\s]?[A-Z])?[-\s]?\d{2,4}[A-Z]?\b")
SEMESTER_RE = re.compile(r"\b(?:sem(?:ester)?|batch|scheme|syllabus|programme outcome|course outcome)\b", re.I)
NOTICE_RE = re.compile(r"\b(?:notice|notification|admission|result|date sheet|timetable|selection list|eligible list)\b", re.I)
CONTACT_RE = re.compile(r"\b(?:email|phone|mobile|contact|faculty|staff|dean|hod|coordinator)\b", re.I)
HEADING_RE = re.compile(
    r"^(?:"
    r"[A-Z][A-Za-z0-9&/().,'’\- ]{2,90}|"
    r"\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9&/().,'’\- ]{2,90}|"
    r"(?:Chapter|Section|Unit|Module|Semester|Batch)\s+[\w .:/()-]{1,60}"
    r")$",
    re.I,
)
SIZES = {
    "pdf": (1250, 220),
    "docx": (1050, 180),
    "html": (1050, 180),
    "htm": (1050, 180),
    "csv": (850, 120),
    "xlsx": (850, 120),
    "txt": (1000, 160),
    "md": (1000, 160),
    "json": (800, 120),
}
SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""]
MIN_CHUNK_LEN = 120
MAX_HEADING_CONTEXT = 4


@dataclass
class SemanticUnit:
    text: str
    kind: str = "paragraph"
    heading_path: tuple[str, ...] = ()
    priority: int = 0


def _splitter(file_type: str) -> RecursiveCharacterTextSplitter:
    chunk_size, chunk_overlap = SIZES.get(file_type, (800, 120))
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )


def _merge_tiny(parts: list[str], min_len: int = MIN_CHUNK_LEN) -> list[str]:
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


def _clean_lines(text: str) -> list[str]:
    lines = []
    for raw in PAGE_BREAK_RE.sub("\n--- PAGE BREAK ---\n", text or "").splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if line:
            lines.append(line)
    return lines


def _is_heading(line: str, next_line: str = "") -> bool:
    line = line.strip(" :-")
    if len(line) < 3 or len(line) > 120:
        return False
    if TABLE_LINE_RE.search(line) or EMAIL_RE.search(line) or URL_RE.search(line):
        return False
    if line.endswith(".") and len(line.split()) > 6:
        return False
    if SEMESTER_RE.search(line) and len(line.split()) <= 10:
        return True
    if line.isupper() and 2 <= len(line.split()) <= 12:
        return True
    if HEADING_RE.match(line) and (not next_line or not line.endswith(",")):
        return True
    return False


def _kind_for(text: str) -> str:
    if _table_row_count(text):
        return "table"
    if COURSE_CODE_RE.search(text) or SEMESTER_RE.search(text):
        return "syllabus"
    if CONTACT_RE.search(text) or EMAIL_RE.search(text) or PHONE_RE.search(text):
        return "contact"
    if NOTICE_RE.search(text):
        return "notice"
    return "paragraph"


def _priority_for(kind: str, text: str) -> int:
    score = 0
    if kind in {"table", "syllabus", "contact", "notice"}:
        score += 2
    if URL_RE.search(text) or EMAIL_RE.search(text) or PHONE_RE.search(text):
        score += 1
    if COURSE_CODE_RE.search(text):
        score += 1
    return score


def _semantic_units(text: str) -> list[SemanticUnit]:
    lines = _clean_lines(text)
    units: list[SemanticUnit] = []
    headings: list[str] = []
    buffer: list[str] = []

    def flush(kind: str | None = None):
        nonlocal buffer
        if not buffer:
            return
        block = "\n".join(buffer).strip()
        buffer = []
        if not block:
            return
        detected = kind or _kind_for(block)
        units.append(
            SemanticUnit(
                text=block,
                kind=detected,
                heading_path=tuple(headings[-MAX_HEADING_CONTEXT:]),
                priority=_priority_for(detected, block),
            )
        )

    for idx, line in enumerate(lines):
        if line == "--- PAGE BREAK ---":
            flush()
            if headings and not headings[-1].lower().startswith("page break"):
                headings = headings[:1]
            continue

        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        if _is_heading(line, next_line):
            flush()
            heading = line.strip()
            if heading not in headings:
                if len(heading.split()) <= 3 and headings:
                    headings = headings[:1] + [heading]
                else:
                    headings.append(heading)
            continue

        if TABLE_LINE_RE.search(line):
            if buffer and _kind_for("\n".join(buffer)) != "table":
                flush()
            buffer.append(line)
            continue

        current_kind = _kind_for("\n".join(buffer)) if buffer else ""
        line_kind = _kind_for(line)
        if buffer and current_kind in {"table", "syllabus", "contact"} and line_kind != current_kind:
            flush(current_kind)
        buffer.append(line)

    flush()
    return _merge_related_units(units)


def _merge_related_units(units: list[SemanticUnit]) -> list[SemanticUnit]:
    merged: list[SemanticUnit] = []
    for unit in units:
        if not unit.text.strip():
            continue
        if (
            merged
            and len(unit.text) < 90
            and unit.kind == merged[-1].kind
            and unit.heading_path == merged[-1].heading_path
        ):
            previous = merged[-1]
            text = f"{previous.text}\n{unit.text}".strip()
            merged[-1] = SemanticUnit(
                text=text,
                kind=previous.kind,
                heading_path=previous.heading_path,
                priority=max(previous.priority, unit.priority),
            )
        else:
            merged.append(unit)
    return merged


def _unit_text(unit: SemanticUnit) -> str:
    heading = " > ".join(unit.heading_path)
    if heading and heading.lower() not in unit.text[:200].lower():
        return f"Section: {heading}\n{unit.text}"
    return unit.text


def _tail_overlap(text: str, target_chars: int) -> str:
    if len(text) <= target_chars:
        return text
    sentences = SENTENCE_SPLIT_RE.split(text.replace("\n", " "))
    tail = []
    total = 0
    for sentence in reversed(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        tail.append(sentence)
        total += len(sentence) + 1
        if total >= target_chars:
            break
    if tail:
        return " ".join(reversed(tail))[-target_chars:].strip()
    return text[-target_chars:].strip()


def _split_large_unit(unit: SemanticUnit, file_type: str) -> list[SemanticUnit]:
    chunk_size, _ = SIZES.get(file_type, (1000, 160))
    if len(unit.text) <= chunk_size * 1.25:
        return [unit]

    if unit.kind == "table":
        lines = unit.text.splitlines()
        pieces = []
        current = []
        for line in lines:
            if current and sum(len(x) + 1 for x in current) + len(line) > chunk_size:
                pieces.append("\n".join(current))
                current = current[:1] if current else []
            current.append(line)
        if current:
            pieces.append("\n".join(current))
    else:
        pieces = _splitter(file_type).split_text(unit.text)

    return [
        SemanticUnit(
            text=piece.strip(),
            kind=unit.kind,
            heading_path=unit.heading_path,
            priority=unit.priority,
        )
        for piece in pieces
        if piece.strip()
    ]


def _pack_semantic_chunks(units: list[SemanticUnit], file_type: str) -> list[str]:
    chunk_size, chunk_overlap = SIZES.get(file_type, (1000, 160))
    expanded: list[SemanticUnit] = []
    for unit in units:
        expanded.extend(_split_large_unit(unit, file_type))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    current_kind = ""

    def flush():
        nonlocal current, current_len, current_kind
        if not current:
            return
        chunks.append("\n\n".join(current).strip())
        current = []
        current_len = 0
        current_kind = ""

    for unit in expanded:
        text = _unit_text(unit)
        unit_len = len(text)
        hard_boundary = (
            current
            and unit.priority >= 2
            and current_kind
            and unit.kind != current_kind
            and current_len >= MIN_CHUNK_LEN
        )
        if current and (current_len + unit_len + 2 > chunk_size or hard_boundary):
            previous = "\n\n".join(current).strip()
            flush()
            overlap = _tail_overlap(previous, chunk_overlap)
            if overlap and unit.kind not in {"table", "contact"}:
                current.append(f"Context from previous chunk:\n{overlap}")
                current_len = len(current[0])
        current.append(text)
        current_len += unit_len + 2
        current_kind = current_kind or unit.kind

    flush()
    return _merge_tiny(chunks)


def _semantic_split(text: str, file_type: str) -> list[str]:
    units = _semantic_units(text)
    if not units:
        return _merge_tiny(_splitter(file_type).split_text(text))
    return _pack_semantic_chunks(units, file_type)


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
    if doc.get("source_kind"):
        header.append(f"Source kind: {doc['source_kind']}")

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

        parts = _semantic_split(text, doc.get("file_type", "txt"))
        for index, part in enumerate(parts):
            chunk_links = _chunk_links(part, doc.get("links", []))
            chunk_text = _format_chunk_text(doc, part)
            table_rows = _table_row_count(part)
            contact_fields = _contact_field_count(part)
            semantic_kind = _kind_for(part)
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
                    "chunk_strategy": "semantic",
                    "semantic_kind": semantic_kind,
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
