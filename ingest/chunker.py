import hashlib
import json
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

# Upgraded SPACE_TABLE_RE: stricter — requires 3+ columns and consistent spacing
# Prevents false matches on normal sentences with double spaces
SPACE_TABLE_RE = re.compile(
    r"^(?:\S[\w.,()%/\-]*\s{2,}){2,}\S[\w.,()%/\-]*\s*$"
)

PAGE_BREAK_RE = re.compile(r"\n\s*--- PAGE BREAK ---\s*\n", re.I)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,8}(?:[-\s]?[A-Z])?[-\s]?\d{2,4}[A-Z]?\b")
SEMESTER_RE = re.compile(
    r"\b(?:sem(?:ester)?|batch|scheme|syllabus|programme outcome|course outcome)\b", re.I
)
NOTICE_RE = re.compile(
    r"\b(?:notice|notification|admission|result|date sheet|timetable|selection list|eligible list)\b",
    re.I,
)
CONTACT_RE = re.compile(
    r"\b(?:email|phone|mobile|contact|faculty|staff|dean|hod|coordinator)\b", re.I
)
HEADING_RE = re.compile(
    r"^(?:"
    r"[A-Z][A-Za-z0-9&/().,''\- ]{2,90}|"
    r"\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9&/().,''\- ]{2,90}|"
    r"(?:Chapter|Section|Unit|Module|Semester|Batch)\s+[\w .:/()-]{1,60}"
    r")$",
    re.I,
)

# Looks like a header row: mostly non-numeric tokens, no sentence-ending punctuation
HEADER_ROW_RE = re.compile(r"^(?:[A-Za-z%.()\-/# ]{2,25}\s{2,}){1,}\S.*$")

SIZES = {
    "pdf":  (1250, 220),
    "docx": (1050, 180),
    "html": (1050, 180),
    "htm":  (1050, 180),
    "csv":  (850,  120),
    "xlsx": (850,  120),
    "txt":  (1000, 160),
    "md":   (1000, 160),
    "json": (800,  120),
}
SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""]
MIN_CHUNK_LEN = 120
MAX_HEADING_CONTEXT = 4

# JSON splitting constants
JSON_LIST_CHUNK_SIZE = 10
JSON_DICT_CHUNK_SIZE = 5

# Contact chunk merging: don't merge chunks larger than this
CONTACT_MERGE_MAX = 160


@dataclass
class SemanticUnit:
    text: str
    kind: str = "paragraph"
    heading_path: tuple = ()
    priority: int = 0


# ---------------------------------------------------------------------------
# Space-aligned table helpers
# ---------------------------------------------------------------------------

def _detect_column_positions(header_line: str) -> list[int]:
    """
    Given a header line like:
        "Name         Dept       Score"
    Return the start positions of each column: [0, 13, 23]
    """
    positions = []
    i = 0
    in_word = False
    for i, ch in enumerate(header_line):
        if ch != " " and not in_word:
            positions.append(i)
            in_word = True
        elif ch == " ":
            in_word = False
    return positions


def _parse_space_table_row(line: str, col_positions: list[int]) -> list[str]:
    """
    Slice a fixed-width row at known column positions.
    Returns list of cell strings.
    """
    cells = []
    for idx, start in enumerate(col_positions):
        end = col_positions[idx + 1] if idx + 1 < len(col_positions) else len(line)
        cell = line[start:end].strip()
        cells.append(cell)
    return cells


def _is_likely_header_row(line: str) -> bool:
    """
    True if a space-aligned line looks like column headers
    (mostly alphabetic tokens, no long numbers, no sentence endings).
    """
    if line.endswith(".") or line.endswith(":"):
        return False
    tokens = line.split()
    if not tokens:
        return False
    alpha_count = sum(1 for t in tokens if re.match(r"^[A-Za-z%.()\-/#]+$", t))
    return alpha_count / len(tokens) >= 0.6


def _format_space_table(lines: list[str]) -> str:
    """
    Convert a block of space-aligned table lines into labelled text:
        Name: Alice | Department: CS | Score: 95.2
    Falls back to pipe-joined if header detection fails.
    """
    if not lines:
        return ""

    # Find header row — first line that looks like headers
    header_idx = 0
    for i, line in enumerate(lines[:3]):
        if _is_likely_header_row(line):
            header_idx = i
            break

    header_line = lines[header_idx]
    col_positions = _detect_column_positions(header_line)
    headers = _parse_space_table_row(header_line, col_positions)

    if len(headers) < 2:
        # Can't parse columns — just return pipe-joined lines
        return "\n".join(" | ".join(line.split()) for line in lines)

    # Separator line (dashes) — skip
    data_start = header_idx + 1
    if data_start < len(lines) and re.match(r"^[-\s]+$", lines[data_start]):
        data_start += 1

    rows_text = []
    # Include header as first labelled line too
    rows_text.append(" | ".join(f"{h}" for h in headers if h))

    for line in lines[data_start:]:
        if not line.strip():
            continue
        cells = _parse_space_table_row(line, col_positions)
        pairs = []
        for header, cell in zip(headers, cells):
            if cell and header:
                pairs.append(f"{header}: {cell}")
            elif cell:
                pairs.append(cell)
        if pairs:
            rows_text.append(" | ".join(pairs))

    return "\n".join(rows_text)


# ---------------------------------------------------------------------------
# JSON-aware splitting
# ---------------------------------------------------------------------------

def _split_json_list(data: list, source_key: str = "") -> list[str]:
    """Split a JSON list into chunks of JSON_LIST_CHUNK_SIZE items each."""
    chunks = []
    for i in range(0, len(data), JSON_LIST_CHUNK_SIZE):
        batch = data[i:i + JSON_LIST_CHUNK_SIZE]
        prefix = f"// {source_key} (items {i + 1}–{i + len(batch)})\n" if source_key else ""
        chunks.append(prefix + json.dumps(batch, ensure_ascii=False, indent=2))
    return chunks


def _split_json_dict(data: dict) -> list[str]:
    """Split a JSON dict by top-level keys into groups of JSON_DICT_CHUNK_SIZE."""
    items = list(data.items())
    chunks = []
    for i in range(0, len(items), JSON_DICT_CHUNK_SIZE):
        batch = dict(items[i:i + JSON_DICT_CHUNK_SIZE])
        chunks.append(json.dumps(batch, ensure_ascii=False, indent=2))
    return chunks


def _split_json_text(text: str) -> list[str] | None:
    """
    Try to parse text as JSON and split it intelligently.
    Returns list of chunk strings, or None if not valid JSON.
    """
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if isinstance(data, list):
        if len(stripped) <= 1200:
            return [stripped]
        return _split_json_list(data)

    if isinstance(data, dict):
        if len(stripped) <= 1200:
            return [stripped]
        chunks = []
        small_keys: dict = {}
        for key, val in data.items():
            if isinstance(val, list) and len(json.dumps(val)) > 600:
                if small_keys:
                    chunks.append(json.dumps(small_keys, ensure_ascii=False, indent=2))
                    small_keys = {}
                sub_chunks = _split_json_list(val, source_key=key)
                chunks.extend(sub_chunks)
            else:
                small_keys[key] = val
                if len(small_keys) >= JSON_DICT_CHUNK_SIZE:
                    chunks.append(json.dumps(small_keys, ensure_ascii=False, indent=2))
                    small_keys = {}
        if small_keys:
            chunks.append(json.dumps(small_keys, ensure_ascii=False, indent=2))
        return chunks if chunks else [stripped]

    return [stripped]


# ---------------------------------------------------------------------------
# Splitter / helpers
# ---------------------------------------------------------------------------

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


def _is_pipe_table_line(line: str) -> bool:
    """Strict pipe-table detection."""
    return bool(TABLE_LINE_RE.search(line))


def _is_space_table_line(line: str) -> bool:
    """
    Detect space-aligned table row.
    Requires: 3+ tokens separated by 2+ spaces, no sentence punctuation.
    """
    if line.endswith(".") or line.endswith("?") or line.endswith(":"):
        return False
    # Must have at least 2 double-space gaps (= 3+ columns)
    gaps = len(re.findall(r"\s{2,}", line))
    if gaps < 2:
        return False
    return bool(SPACE_TABLE_RE.match(line))


def _is_table_line(line: str) -> bool:
    """Combined table line detection."""
    return _is_pipe_table_line(line) or _is_space_table_line(line)


def _table_type(line: str) -> str:
    """Return 'pipe' or 'space' for table line type."""
    if _is_pipe_table_line(line):
        return "pipe"
    if _is_space_table_line(line):
        return "space"
    return "none"


def _kind_for(text: str) -> str:
    stripped = text.strip()
    # Structured JSON data
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return "structured"
        except json.JSONDecodeError:
            pass
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
    if kind in {"table", "syllabus", "contact", "notice", "structured"}:
        score += 2
    if URL_RE.search(text) or EMAIL_RE.search(text) or PHONE_RE.search(text):
        score += 1
    if COURSE_CODE_RE.search(text):
        score += 1
    return score


# ---------------------------------------------------------------------------
# Semantic unit building
# ---------------------------------------------------------------------------

def _semantic_units(text: str) -> list[SemanticUnit]:
    lines = _clean_lines(text)
    units: list[SemanticUnit] = []
    headings: list[str] = []
    buffer: list[str] = []
    current_table_type: str = "none"  # track pipe vs space tables separately

    def flush(kind: str | None = None):
        nonlocal buffer, current_table_type
        if not buffer:
            return
        block = "\n".join(buffer).strip()
        buffer = []
        current_table_type = "none"
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

    def flush_space_table():
        """Flush a space-aligned table buffer with column labelling applied."""
        nonlocal buffer, current_table_type
        if not buffer:
            return
        formatted = _format_space_table(buffer)
        buffer = []
        current_table_type = "none"
        if not formatted.strip():
            return
        units.append(
            SemanticUnit(
                text=formatted,
                kind="table",
                heading_path=tuple(headings[-MAX_HEADING_CONTEXT:]),
                priority=_priority_for("table", formatted),
            )
        )

    for idx, line in enumerate(lines):
        if line == "--- PAGE BREAK ---":
            if current_table_type == "space":
                flush_space_table()
            else:
                flush()
            if headings and not headings[-1].lower().startswith("page break"):
                headings = headings[:1]
            continue

        next_line = lines[idx + 1] if idx + 1 < len(lines) else ""

        if _is_heading(line, next_line):
            if current_table_type == "space":
                flush_space_table()
            else:
                flush()
            heading = line.strip()
            if heading not in headings:
                if len(heading.split()) <= 3 and headings:
                    headings = headings[:1] + [heading]
                else:
                    headings.append(heading)
            continue

        ttype = _table_type(line)

        if ttype == "pipe":
            # Flush any space-table in progress before starting pipe table
            if current_table_type == "space":
                flush_space_table()
            elif buffer and _kind_for("\n".join(buffer)) != "table":
                flush()
            current_table_type = "pipe"
            buffer.append(line)
            continue

        if ttype == "space":
            # Flush any pipe-table or non-table buffer
            if current_table_type == "pipe":
                flush()
            elif buffer and _kind_for("\n".join(buffer)) != "table":
                flush()
            current_table_type = "space"
            buffer.append(line)
            continue

        # Non-table line
        if current_table_type == "space":
            flush_space_table()
        elif current_table_type == "pipe":
            flush()

        current_kind = _kind_for("\n".join(buffer)) if buffer else ""
        line_kind = _kind_for(line)
        if buffer and current_kind in {"table", "syllabus", "contact", "structured"} and line_kind != current_kind:
            flush(current_kind)
        buffer.append(line)

    # Final flush
    if current_table_type == "space":
        flush_space_table()
    else:
        flush()

    return _merge_related_units(units)


def _merge_related_units(units: list[SemanticUnit]) -> list[SemanticUnit]:
    merged: list[SemanticUnit] = []
    for unit in units:
        if not unit.text.strip():
            continue

        # Upgraded: don't merge contact chunks that already have enough info
        if (
            merged
            and unit.kind == "contact"
            and len(unit.text) > CONTACT_MERGE_MAX
        ):
            merged.append(unit)
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


# ---------------------------------------------------------------------------
# Large unit splitting
# ---------------------------------------------------------------------------

def _split_large_unit(unit: SemanticUnit, file_type: str) -> list[SemanticUnit]:
    chunk_size, _ = SIZES.get(file_type, (1000, 160))
    if len(unit.text) <= chunk_size * 1.25:
        return [unit]

    if unit.kind == "table":
        lines = unit.text.splitlines()
        # Keep up to 2 header rows, re-attach to every sub-chunk
        header_rows = _find_header_rows(lines)
        pieces = []
        current = header_rows[:]
        for line in lines[len(header_rows):]:
            projected_len = sum(len(x) + 1 for x in current) + len(line)
            if current and projected_len > chunk_size:
                pieces.append("\n".join(current))
                current = header_rows[:]
            current.append(line)
        if current and current != header_rows:
            pieces.append("\n".join(current))

    elif unit.kind == "structured":
        json_chunks = _split_json_text(unit.text)
        if json_chunks:
            return [
                SemanticUnit(
                    text=piece.strip(),
                    kind="structured",
                    heading_path=unit.heading_path,
                    priority=unit.priority,
                )
                for piece in json_chunks
                if piece.strip()
            ]
        pieces = _splitter(file_type).split_text(unit.text)

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


def _find_header_rows(lines: list[str]) -> list[str]:
    """
    Identify up to 2 header rows from a table's lines.
    A header row is one that looks like column labels
    (mostly alphabetic, OR is a separator line of dashes).
    """
    headers = []
    for line in lines[:3]:
        if _is_likely_header_row(line) or re.match(r"^[-| ]+$", line):
            headers.append(line)
        else:
            break
    return headers if headers else lines[:1]


# ---------------------------------------------------------------------------
# Packing into final chunks
# ---------------------------------------------------------------------------

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
        # Never mix structured/table with other kinds
        kind_boundary = (
            current
            and current_kind in {"structured", "table"}
            and unit.kind != current_kind
        ) or (
            current
            and unit.kind in {"structured", "table"}
            and current_kind != unit.kind
        )

        if current and (
            current_len + unit_len + 2 > chunk_size
            or hard_boundary
            or kind_boundary
        ):
            previous = "\n\n".join(current).strip()
            flush()
            overlap = _tail_overlap(previous, chunk_overlap)
            if overlap:
                current.append(f"Context from previous chunk:\n{overlap}")
                current_len = len(current[0])

        current.append(text)
        current_len += unit_len + 2
        current_kind = current_kind or unit.kind

    flush()
    return _merge_tiny(chunks)


# ---------------------------------------------------------------------------
# Top-level split entry point
# ---------------------------------------------------------------------------

def _semantic_split(text: str, file_type: str) -> list[str]:
    # JSON files: JSON-aware splitting first
    if file_type == "json":
        json_chunks = _split_json_text(text)
        if json_chunks:
            return _merge_tiny(json_chunks)

    units = _semantic_units(text)
    if not units:
        return _merge_tiny(_splitter(file_type).split_text(text))
    return _pack_semantic_chunks(units, file_type)


# ---------------------------------------------------------------------------
# Link extraction
# ---------------------------------------------------------------------------

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
    return sum(1 for line in text.splitlines() if _is_table_line(line))


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
                    "is_structured": semantic_kind == "structured",
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
    print("By semantic kind:", dict(Counter(item["semantic_kind"] for item in chunks)))
    print(f"Structured chunks: {sum(1 for c in chunks if c['is_structured'])}")
    print(f"Table chunks:      {sum(1 for c in chunks if c['has_table'])}")
    print(f"Contact chunks:    {sum(1 for c in chunks if c['contact_field_count'] > 0)}")
    print(f"With links:        {sum(1 for c in chunks if c['has_links'])}")
    for link in find_links_in_chunks(chunks, "admission form cuet ug")[:5]:
        print(f"  {link['anchor_text']} -> {link['url']}")