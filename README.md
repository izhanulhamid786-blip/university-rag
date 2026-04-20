# University RAG

A production-ready university RAG assistant that crawls official university data, indexes documents and tables, and answers student and staff queries with source-grounded responses.

## What It Does

- Crawls a university website, including SPA routes and linked documents
- Extracts content from HTML, PDF, DOCX, XLSX, TXT, CSV, JSON, and PPTX files
- Preserves table-like data for better staff directories, contact records, and count queries
- Builds a hybrid retrieval pipeline using Chroma vector search and BM25
- Improves ranking with a cross-encoder reranker
- Resolves follow-up questions like pronoun-based references in chat history
- Serves a Streamlit chat app with grounded answers and source links

## Stack

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- BM25
- Google Gemini API
- Playwright

## Project Structure

```text
app.py                Streamlit chat application
crawler.py            Website crawler and document extractor
ingest/loader.py      Loads crawler/manual data into normalized documents
ingest/chunker.py     Splits documents into retrieval chunks
ingest/build_db.py    Builds the Chroma vector database
rag/retriever.py      Hybrid retrieval and heuristic ranking
rag/reranker.py       Cross-encoder reranking
rag/memory.py         Follow-up question rewriting
rag/pipeline.py       End-to-end retrieval and generation pipeline
rag/prompt.py         Grounded answer prompt builder
data/structured/      Structured crawler output used by the RAG pipeline
data/manual/          Optional manually added files for indexing
vector_db/            Local Chroma database
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
python -m playwright install chromium
```

3. Create your environment file.

```powershell
Copy-Item .env.example .env
```

4. Add your Gemini API key to `.env`.

```env
GOOGLE_API_KEY=your_key_here
```

## Environment Variables

Key runtime settings are defined in `.env.example`.

- `GOOGLE_API_KEY`: Gemini API key
- `GENERATOR_MODEL`: main answer generation model
- `FALLBACK_GENERATOR_MODEL`: fallback generation model
- `RAG_EMBED_MODEL`: embedding model for vector search
- `RAG_RERANK_MODEL`: cross-encoder reranker
- `RAG_RETRIEVAL_K`: number of retrieved candidates
- `RAG_RERANK_TOP_K`: final number of chunks passed into prompt building
- `RAG_RERANK_CANDIDATES`: number of candidates sent to reranking

## Run The Project

If you already have crawler output and an existing vector DB:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

For a full refresh:

1. Crawl the site

```powershell
python crawler.py
```

2. Rebuild the vector database

```powershell
python -m ingest.build_db --reset
```

3. Start the app

```powershell
streamlit run app.py
```

## Data Notes

- `data/structured/` is the primary source used for retrieval
- `data/manual/` can be used to add curated files
- `data/` root files are included in retrieval by default so locally added PDFs and other raw files are searchable
- `vector_db/` is generated locally and should not be committed

## Current Retrieval Features

- Hybrid dense + sparse retrieval
- Staff/contact-aware ranking boosts
- Better handling of directory-style tables
- Follow-up question rewriting for conversational queries
- Source deduplication and grounded response generation

## Recommended Workflow

```powershell
.\venv\Scripts\Activate.ps1
python crawler.py
python -m ingest.build_db --reset
streamlit run app.py
```

## Public Repo Checklist

- Keep `.env` private
- Rebuild `vector_db/` locally, not in Git
- Review raw files under `data/` before publishing them
- Update the repository description on GitHub for better discoverability

## License

No license has been added yet.
