# Central University of Kashmir RAG

Production-hardened university assistant with:

- Playwright crawler for SPA content and documents
- Clean ingestion from crawler output
- Hybrid retrieval with Chroma + BM25
- Cross-encoder reranking
- Streamlit chat app with source links

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
playwright install chromium
```

3. Copy `.env.example` to `.env` and add `GOOGLE_API_KEY`.
   The default generator is `gemini-2.5-pro` for answer quality. If you want lower latency and cost, override it with `GENERATOR_MODEL=gemini-2.5-flash`.

## Data Flow

1. Crawl the university site:

```powershell
python crawler.py
```

2. Build or rebuild the vector database:

```powershell
python -m ingest.build_db --reset
```

3. Start the app:

```powershell
streamlit run app.py
```

## Data Sources

- `data/structured/`: primary crawler output used by the production RAG pipeline
- `data/manual/`: curated extra files you want indexed
- `data/`: legacy raw files are excluded by default to avoid noisy retrieval

## Publish Checklist

- Set `GOOGLE_API_KEY` in `.env`
- Run the crawler until `data/structured/` is populated with clean records
- Rebuild the vector DB with `python -m ingest.build_db --reset`
- Verify the sidebar in the app reports a ready knowledge base
