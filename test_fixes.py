import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import run_with_metadata
from rag.memory import rewrite_query

print("=== TEST 1: Dr. Yash Paul ===")
history1 = [
    {"role": "user", "content": "who is dr yash paul"},
    {"role": "assistant", "content": "Dr. Yash Paul is an Assistant Professor and Coordinator at the Department of IT."}
]
query1 = "whats his phne number"
rewritten1 = rewrite_query(query1, history1)
print(f"Rewritten: {rewritten1}")
stream, sources, details = run_with_metadata(query1, history1)
answer = "".join(getattr(chunk, "text", "") for chunk in stream)
print(f"Answer: {answer}")
print(f"Sources: {len(sources)}")
print(f"Rerank Score: {sources[0]['rerank_score'] if sources else 'N/A'}")

print("\n=== TEST 2: Afaq Alam Khan ===")
history2 = [
    {"role": "user", "content": "who is afaq alam khan"},
    {"role": "assistant", "content": "Dr. Afaq Alam Khan is the Dean, Students Welfare at the Central University of Kashmir."}
]
query2 = "no he is an assistant professor"
rewritten2 = rewrite_query(query2, history2)
print(f"Rewritten: {rewritten2}")
stream, sources, details = run_with_metadata(query2, history2)
answer = "".join(getattr(chunk, "text", "") for chunk in stream)
print(f"Answer: {answer}")
