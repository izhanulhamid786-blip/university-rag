import argparse
import csv
import importlib
import inspect
import json
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.pipeline import run_with_metadata
from rag.model_loading import clear_broken_proxy_env
from rag.settings import get_settings


DEFAULT_DATASET = Path("evals/questions.example.jsonl")
DEFAULT_OUTPUT_DIR = Path("evals/results")


@dataclass
class EvalRow:
    question: str
    reference: str | None = None
    history: list[dict] | None = None
    answer_style: str = "balanced"


def _clean_reference(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    reference = value.strip()
    if not reference:
        return None
    if reference.lower().startswith("replace this with"):
        return None
    return reference


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
    return rows


def load_eval_rows(path: Path) -> list[EvalRow]:
    rows = []
    for item in _read_jsonl(path):
        question = (item.get("question") or item.get("user_input") or "").strip()
        if not question:
            raise ValueError("Every evaluation row needs a 'question' field.")
        reference = item.get("reference") or item.get("ground_truth") or item.get("answer")
        rows.append(
            EvalRow(
                question=question,
                reference=_clean_reference(reference),
                history=item.get("history") or [],
                answer_style=item.get("answer_style") or "balanced",
            )
        )
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}.")
    return rows


def _collect_text(stream: Any) -> str:
    parts = []
    for item in stream:
        text = getattr(item, "text", "")
        if text:
            parts.append(text)
    return "".join(parts).strip()


def run_rag_samples(rows: list[EvalRow]) -> list[dict]:
    samples = []
    for index, row in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {row.question}")
        stream, sources, metadata = run_with_metadata(
            row.question,
            row.history or [],
            answer_style=row.answer_style,
        )
        response = _collect_text(stream)
        contexts = metadata.get("retrieved_contexts") or [source.get("preview", "") for source in sources]
        samples.append(
            {
                "user_input": row.question,
                "response": response,
                "retrieved_contexts": [context for context in contexts if context],
                "reference": row.reference,
                "source_count": len(sources),
                "candidate_count": metadata.get("candidate_count", 0),
                "selected_chunk_count": metadata.get("selected_chunk_count", 0),
                "timings_ms": metadata.get("timings_ms", {}),
            }
        )
    return samples


def _ragas_imports():
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
    except ImportError:
        from ragas import EvaluationDataset, evaluate
        from ragas.dataset_schema import SingleTurnSample
    return EvaluationDataset, SingleTurnSample, evaluate


def _metric_instances(has_references: bool) -> list[Any]:
    metrics_module = importlib.import_module("ragas.metrics")

    def metric(*names: str):
        for name in names:
            value = getattr(metrics_module, name, None)
            if value is None:
                continue
            return value() if inspect.isclass(value) else value
        raise ImportError(f"None of these RAGAS metrics are available: {', '.join(names)}")

    metrics = [
        metric("Faithfulness", "faithfulness"),
        metric("ResponseRelevancy", "answer_relevancy"),
    ]
    if has_references:
        metrics.extend(
            [
                metric("LLMContextPrecisionWithReference", "context_precision"),
                metric("LLMContextRecall", "context_recall"),
                metric("FactualCorrectness", "AnswerCorrectness", "answer_correctness"),
            ]
        )
    return metrics


def _ragas_llm():
    return None


def _ragas_embeddings():
    settings = get_settings()
    clear_broken_proxy_env()
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        return None
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.embed_model,
            model_kwargs={"local_files_only": settings.local_files_only},
            encode_kwargs={"normalize_embeddings": True},
        )
    )


def evaluate_samples(
    samples: list[dict],
    *,
    timeout: int = 600,
    max_workers: int = 1,
    batch_size: int | None = 1,
):
    EvaluationDataset, SingleTurnSample, evaluate = _ragas_imports()
    from ragas.run_config import RunConfig

    has_references = all(sample.get("reference") for sample in samples)
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=sample["user_input"],
                response=sample["response"],
                retrieved_contexts=sample["retrieved_contexts"],
                reference=sample.get("reference"),
            )
            for sample in samples
        ]
    )
    kwargs = {
        "dataset": dataset,
        "metrics": _metric_instances(has_references),
        "run_config": RunConfig(timeout=timeout, max_workers=max_workers),
        "batch_size": batch_size,
    }
    llm = _ragas_llm()
    embeddings = _ragas_embeddings()
    if llm is not None:
        kwargs["llm"] = llm
    if embeddings is not None:
        kwargs["embeddings"] = embeddings
    return evaluate(**kwargs)


def _result_rows(result: Any, samples: list[dict]) -> list[dict]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        rows = frame.to_dict(orient="records")
    else:
        rows = [dict(result)]
    for index, row in enumerate(rows):
        if index < len(samples):
            row.setdefault("user_input", samples[index]["user_input"])
            row.setdefault("response", samples[index]["response"])
            row.setdefault("reference", samples[index].get("reference"))
            row.setdefault("source_count", samples[index]["source_count"])
            row.setdefault("candidate_count", samples[index]["candidate_count"])
            row.setdefault("selected_chunk_count", samples[index]["selected_chunk_count"])
    return rows


def save_outputs(samples: list[dict], result: Any, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_path = output_dir / f"ragas-samples-{stamp}.json"
    csv_path = output_dir / f"ragas-results-{stamp}.csv"
    raw_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = _result_rows(result, samples)
    fieldnames = sorted({field for row in rows for field in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return raw_path, csv_path


def main() -> int:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"langchain.*")

    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline with RAGAS.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="JSONL file with question/reference rows.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for result files.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-job RAGAS timeout in seconds.")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel RAGAS judge workers.")
    parser.add_argument("--batch-size", type=int, default=1, help="RAGAS batch size.")
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face model downloads during evaluation. By default, cached local models are used.",
    )
    args = parser.parse_args()

    clear_broken_proxy_env()
    if not args.allow_downloads:
        os.environ.setdefault("RAG_LOCAL_FILES_ONLY", "true")

    try:
        rows = load_eval_rows(args.dataset)
        samples = run_rag_samples(rows)
        result = evaluate_samples(
            samples,
            timeout=args.timeout,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        message = str(exc)
        if "local_files_only" in message or "huggingface" in message.lower() or "hf_hub" in message.lower():
            print(
                "\nEvaluation could not load a local Hugging Face model.\n"
                "Run once with downloads enabled:\n"
                "  python -m evals.ragas_eval --dataset evals/questions.example.jsonl --allow-downloads\n"
                "Or make sure the embedding and reranker models are already cached locally."
            )
            print(f"\nDetails: {exc}")
            return 1
        raise
    raw_path, csv_path = save_outputs(samples, result, args.output_dir)

    print("\nRAGAS summary:")
    print(result)
    print(f"\nSaved raw samples: {raw_path}")
    print(f"Saved scored CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
