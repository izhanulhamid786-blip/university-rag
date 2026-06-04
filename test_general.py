import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import run_with_metadata

queries = [
    "how much land does cuk have",
    "who is prof shahid rasool",
    "tell me about the department of physics",
    "who is afaq alam khan"
]

for q in queries:
    print(f"\n\n=== {q} ===")
    stream, sources, details = run_with_metadata(q, [])
    answer = "".join(getattr(chunk, "text", "") for chunk in stream)
    print(f"Answer: {answer}")
    print(f"Top Source Score: {sources[0]['rerank_score'] if sources else 'N/A'}")
