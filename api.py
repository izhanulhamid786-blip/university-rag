import json
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from rag.pipeline import app_status, run_with_metadata, warmup_local_models
from rag.text_cleanup import clean_text_artifacts


class HistoryItem(BaseModel):
    user: str = ""
    bot: str = ""


class QueryRequest(BaseModel):
    query: str = Field(default="", max_length=2000)
    history: list[HistoryItem] = Field(default_factory=list)
    answer_style: str = "balanced"


app = FastAPI(title="University RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    warmup_local_models()

data_dir = Path(__file__).parent / "data"
manual_dir = data_dir / "manual"
pdfs_dir = data_dir / "pdfs"

if manual_dir.exists():
    app.mount("/files/manual", StaticFiles(directory=str(manual_dir)), name="manual")
if pdfs_dir.exists():
    app.mount("/files/pdfs", StaticFiles(directory=str(pdfs_dir)), name="pdfs")


@app.get("/status")
def status() -> dict[str, Any]:
    return app_status()


def _no_info_answer(text: str) -> bool:
    normalized = " ".join(text.split()).strip().lower()
    return (
        "i don't have that information. please contact the university office directly." in normalized
        or "contact the university office directly" in normalized
    )


@app.post("/query")
def query(payload: QueryRequest) -> dict[str, Any]:
    question = payload.query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query is required.")

    history = [item.model_dump() for item in payload.history]
    answer, sources, details = run_with_metadata(
        question,
        history,
        answer_style=payload.answer_style,
    )

    if _no_info_answer(answer):
        sources = []

    return {
        "answer": answer,
        "sources": sources,
        "details": details,
    }

