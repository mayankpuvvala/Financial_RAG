"""
RAGAS evaluation for the Financial RAG system.

Runs a pre-defined test set through the full RAG pipeline, collects
contexts and answers, then scores them using four RAGAS metrics:

  Metric               What it measures
  ─────────────────    ────────────────────────────────────────────
  faithfulness         Is every claim in the answer grounded in the
                       retrieved context (no hallucinations)?
  answer_relevancy     Does the answer actually address the question?
  context_precision    Are the retrieved chunks relevant (not noisy)?
  context_recall       Does context cover all key facts in the ground
                       truth? (requires ground_truth column)

Usage
-----
    # Basic — prints a score table
    python -m evaluation.ragas_eval

    # Save results to JSON
    python -m evaluation.ragas_eval --output results/ragas_scores.json

    # Evaluate only a subset of the test set
    python -m evaluation.ragas_eval --limit 5

Environment
-----------
RAGAS uses the Groq LLM (same as the RAG system) as its judge model.
Make sure GROQ_API_KEY (or groq_api in .env) is set before running.

RAGAS >= 0.2 is required (uses LangchainLLMWrapper + ChatGroq).
Install extras:
    pip install "ragas>=0.2.0" langchain-groq langchain-huggingface
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from query import ask

# ── built-in test set ────────────────────────────────────────────────────────
# Ground-truth figures come directly from the 10-K filings.
# All dollar amounts are in millions unless stated.
_DEFAULT_TEST_SET = [
    {
        "question": "What were Apple total net sales in fiscal year 2024?",
        "ground_truth": (
            "Apple's total net sales for fiscal year 2024 (year ended September 28, 2024) "
            "were $391,035 million, approximately $391 billion. "
            "Products contributed $294,866 million and Services $96,169 million."
        ),
    },
    {
        "question": "What were Microsoft research and development expenses in fiscal year 2024?",
        "ground_truth": (
            "Microsoft's research and development expenses for fiscal year 2024 "
            "(year ended June 30, 2024) were $29,510 million, approximately $29.5 billion, "
            "an increase of 9% from $27,195 million in fiscal year 2023."
        ),
    },
    {
        "question": "What were Alphabet research and development expenses in fiscal year 2024?",
        "ground_truth": (
            "Alphabet's research and development expenses for fiscal year 2024 "
            "(year ended December 31, 2024) were $49,326 million, approximately $49.3 billion, "
            "an increase from $45,427 million in fiscal year 2023."
        ),
    },
    {
        "question": "What was JPMorgan Chase net income in fiscal year 2024?",
        "ground_truth": (
            "JPMorgan Chase's net income for fiscal year 2024 (year ended December 31, 2024) "
            "was $58,471 million, approximately $58.5 billion, "
            "compared to $49,552 million in fiscal year 2023."
        ),
    },
    {
        "question": "What was JPMorgan Chase net income in fiscal year 2023?",
        "ground_truth": (
            "JPMorgan Chase's net income for fiscal year 2023 (year ended December 31, 2023) "
            "was $49,552 million, approximately $49.6 billion, "
            "compared to $37,676 million in fiscal year 2022."
        ),
    },
    {
        "question": "What were Apple operating income and operating margin in fiscal year 2024?",
        "ground_truth": (
            "Apple's operating income for fiscal year 2024 was $123,216 million. "
            "Total net sales were $391,035 million, giving an operating margin of "
            "approximately 31.5%."
        ),
    },
    {
        "question": "What were Microsoft total revenue and net income in fiscal year 2024?",
        "ground_truth": (
            "Microsoft's total revenue for fiscal year 2024 was $245,122 million "
            "(approximately $245 billion), and net income was $88,136 million "
            "(approximately $88 billion)."
        ),
    },
    {
        "question": "What is the current Bitcoin price?",
        "ground_truth": (
            "Bitcoin price information is not available in SEC 10-K filings. "
            "This question is out of scope for the Financial RAG system."
        ),
    },
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_test_set(path: Path | None) -> list[dict]:
    """Load test set from file or return built-in set."""
    if path and path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} test cases from {path}")
        return data
    logger.info(f"Using built-in test set ({len(_DEFAULT_TEST_SET)} questions)")
    return _DEFAULT_TEST_SET


def _run_pipeline(test_set: list[dict], delay: float = 1.5) -> list[dict]:
    """
    Run every question through the RAG pipeline.
    Returns a list of dicts with question/answer/contexts/ground_truth.
    """
    rows: list[dict] = []
    for i, item in enumerate(test_set, 1):
        q = item["question"]
        logger.info(f"[{i}/{len(test_set)}] {q[:80]}")
        try:
            result = ask(q)
            contexts = [rc.chunk.text for rc in result.chunks_used]
            rows.append(
                {
                    "question":     q,
                    "answer":       result.answer,
                    "contexts":     contexts if contexts else ["No context retrieved."],
                    "ground_truth": item.get("ground_truth", ""),
                }
            )
        except Exception as exc:
            logger.warning(f"  Pipeline error: {exc}")
            rows.append(
                {
                    "question":     q,
                    "answer":       f"ERROR: {exc}",
                    "contexts":     [],
                    "ground_truth": item.get("ground_truth", ""),
                }
            )
        if delay and i < len(test_set):
            time.sleep(delay)   # respect Groq free-tier rate limits
    return rows


def _build_ragas_dataset(rows: list[dict]) -> Any:
    """Convert pipeline results into a RAGAS-compatible HuggingFace Dataset."""
    from datasets import Dataset  # type: ignore

    return Dataset.from_list(rows)


def _configure_ragas_llm():
    """Return a RAGAS-compatible LLM wrapper using Groq."""
    try:
        from langchain_groq import ChatGroq                    # type: ignore
        from ragas.llms import LangchainLLMWrapper             # type: ignore

        groq_key = settings.groq_api or os.environ.get("GROQ_API_KEY", "")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=0,
        )
        return LangchainLLMWrapper(llm)
    except ImportError as e:
        logger.error(
            f"Missing dependency: {e}\n"
            "Install with: pip install langchain-groq 'ragas>=0.2.0'"
        )
        raise


def _configure_ragas_embeddings():
    """Return a RAGAS-compatible embeddings wrapper using the local BGE model."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings   # type: ignore
        from ragas.embeddings import LangchainEmbeddingsWrapper    # type: ignore

        hf = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        return LangchainEmbeddingsWrapper(hf)
    except ImportError as e:
        logger.warning(
            f"HuggingFace embeddings not available ({e}). "
            "answer_relevancy metric may fall back to Groq embeddings."
        )
        return None


def _run_ragas(dataset: Any, output: Path | None) -> dict:
    """Run RAGAS evaluation and return scores."""
    from ragas import evaluate                               # type: ignore
    from ragas.metrics import (                              # type: ignore
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    ragas_llm        = _configure_ragas_llm()
    ragas_embeddings = _configure_ragas_embeddings()

    # Attach the judge LLM to each metric
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = ragas_llm
        if ragas_embeddings and hasattr(m, "embeddings"):
            m.embeddings = ragas_embeddings

    logger.info("Running RAGAS evaluation (this may take a few minutes)…")
    result = evaluate(dataset, metrics=metrics)

    scores: dict = {
        "faithfulness":      float(result["faithfulness"]),
        "answer_relevancy":  float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall":    float(result["context_recall"]),
    }

    # ── print table ──────────────────────────────────────────────────────────
    width = 22
    print("\n" + "=" * 50)
    print("  RAGAS Evaluation Results")
    print("=" * 50)
    for k, v in scores.items():
        bar_len = int(v * 20)
        bar     = "#" * bar_len + "." * (20 - bar_len)
        print(f"  {k:<{width}} {v:.4f}  [{bar}]")
    avg = sum(scores.values()) / len(scores)
    print("-" * 50)
    print(f"  {'Average':<{width}} {avg:.4f}")
    print("=" * 50 + "\n")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(
                {"scores": scores, "per_question": result.to_pandas().to_dict(orient="records")},
                f,
                indent=2,
            )
        logger.success(f"Results saved → {output}")

    return scores


# ── entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="RAGAS evaluation for Financial RAG"
    )
    ap.add_argument(
        "--test-set",
        type=Path,
        default=None,
        help="Path to a JSON file with {question, ground_truth} rows "
             "(defaults to built-in set)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save full results to this JSON file",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N questions",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds to wait between RAG calls (default: 1.5 to avoid rate limits)",
    )
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")

    test_set = _load_test_set(args.test_set)
    if args.limit:
        test_set = test_set[: args.limit]

    rows    = _run_pipeline(test_set, delay=args.delay)
    dataset = _build_ragas_dataset(rows)
    _run_ragas(dataset, args.output)


if __name__ == "__main__":
    main()
