"""Diagnostic script to test multiple queries and identify where the RAG pipeline fails."""
import sys, os, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("RAG_GENERATOR_PROVIDER", "gemini")

from rag.retriever import hybrid_retrieve, _dense_search, _bm25_search, _expanded_query
from rag.reranker import rerank
from rag.settings import get_settings

# Test queries covering different types of information
TEST_QUERIES = [
    "Who is Afaq Alam Khan?",
    "What is the date of joining of Arfa Zahoor?",
    "What are the departments in Central University of Kashmir?",
    "What is the admission process for B.Tech CSE?",
    "Who is the Vice Chancellor of CUK?",
    "What is the fee structure for M.Tech?",
    "Contact details of Department of Computer Science",
    "What courses are offered at CUK?",
    "Tell me about the placement cell",
    "What is the eligibility for PhD admission?",
    "How many faculty members are in the IT department?",
    "What is the syllabus for BCA?",
]

def diagnose_query(query: str):
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    
    expanded = _expanded_query(query)
    if expanded != query:
        print(f"  Expanded query: {expanded}")
    
    # Step 1: Dense search
    t0 = time.time()
    dense = _dense_search(query, 80)
    t_dense = time.time() - t0
    
    # Step 2: BM25 search  
    t0 = time.time()
    sparse = _bm25_search(query, 80)
    t_sparse = time.time() - t0
    
    # Step 3: Hybrid retrieve
    t0 = time.time()
    candidates = hybrid_retrieve(query, k=80)
    t_hybrid = time.time() - t0
    
    print(f"\n  [RETRIEVAL] dense={len(dense)} | bm25={len(sparse)} | hybrid={len(candidates)} | time: dense={t_dense:.1f}s bm25={t_sparse:.1f}s hybrid={t_hybrid:.1f}s")
    
    if not candidates:
        print("  *** NO CANDIDATES FOUND - RETRIEVAL FAILURE ***")
        return
    
    # Show top 3 candidates before reranking
    print(f"\n  --- TOP 3 CANDIDATES (before rerank) ---")
    for i, c in enumerate(candidates[:3]):
        title = c.get('title', 'N/A')[:60]
        text_preview = ' '.join(c.get('text', '')[:200].split())
        score = c.get('final_score', 0)
        matched = c.get('matched_by', [])
        print(f"  [{i+1}] score={score:.4f} matched_by={matched} title={title}")
        print(f"       {text_preview}...")
    
    # Step 4: Reranking
    t0 = time.time()
    reranked = rerank(query, candidates, top_k=10)
    t_rerank = time.time() - t0
    
    print(f"\n  [RERANKING] input={len(candidates)} -> output={len(reranked)} | time={t_rerank:.1f}s")
    
    if not reranked:
        print("  *** ALL CANDIDATES PRUNED BY RERANKER ***")
        return
    
    # Show top 3 reranked results
    print(f"\n  --- TOP 3 RERANKED RESULTS ---")
    for i, c in enumerate(reranked[:3]):
        title = c.get('title', 'N/A')[:60]
        text_preview = ' '.join(c.get('text', '')[:200].split())
        rerank_score = c.get('rerank_score', 0)
        rerank_final = c.get('rerank_final_score', 0)
        print(f"  [{i+1}] rerank_score={rerank_score:.4f} final={rerank_final:.4f} title={title}")
        print(f"       {text_preview}...")
    
    # Check if reranker is being too aggressive
    if len(reranked) < 3:
        print(f"\n  *** WARNING: Only {len(reranked)} chunks survived reranking (too aggressive pruning?) ***")
    
    # Show score distribution
    scores = [c.get('rerank_score', 0) for c in reranked]
    print(f"\n  [SCORE DISTRIBUTION] max={max(scores):.4f} min={min(scores):.4f} avg={sum(scores)/len(scores):.4f}")
    
    # Check relevance of top chunk
    top_text = reranked[0].get('text', '')
    query_words = set(query.lower().split())
    found_words = sum(1 for w in query_words if w.lower() in top_text.lower())
    print(f"  [RELEVANCE CHECK] Query words found in top chunk: {found_words}/{len(query_words)}")
    
    return reranked


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    
    settings = get_settings()
    print(f"Settings: embed_model={settings.embed_model}")
    print(f"Settings: rerank_top_k={settings.rerank_top_k}")
    print(f"Settings: max_context_chars={settings.max_context_chars}")
    
    # Check collection status
    from rag.retriever import collection_status
    status = collection_status()
    print(f"Collection: {status}")
    
    results = {}
    for q in TEST_QUERIES:
        try:
            r = diagnose_query(q)
            results[q] = "OK" if r and len(r) >= 3 else f"WEAK ({len(r) if r else 0} results)"
        except Exception as e:
            results[q] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()
    
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for q, status in results.items():
        icon = "✓" if status == "OK" else "✗"
        print(f"  {icon} {status:20s} | {q}")
