"""
Ask a curated question set through a running API's /chat endpoint (local or
deployed) and save the raw turns for review.

    python -m evaluation.chat_probe --base http://127.0.0.1:8000 --out data/eval_results/chat_probe.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# (id, question, filters, follow_up_of)
QUESTIONS = [
    ("aapl-rev",      "What was Apple's total revenue in fiscal 2024?", None, None),
    ("aapl-rev-2023", "And what about 2023?", None, "aapl-rev"),                    # multi-turn follow-up
    ("msft-googl-rd", "Compare Microsoft and Alphabet's R&D spending in 2024.", None, None),
    ("jpm-trend",     "How did JPMorgan's net income change from 2023 to 2025?", None, None),
    ("bac-ni",        "What was Bank of America's net income in fiscal 2024?", None, None),     # restatement-sensitive
    ("amzn-two",      "What were Amazon's operating income and net income in 2024?", None, None),
    ("msft-chip-oi",  "What was the operating income?", {"tickers": ["MSFT"], "years": [2024]}, None),  # UI chips
    ("wfc-risks",     "What are the main risk factors Wells Fargo discloses?", None, None),
    ("msft-segments", "Which business segments does Microsoft report?", None, None),
    ("btc",           "What is the price of Bitcoin today?", None, None),
    ("spacex",        "What was SpaceX's revenue in 2024?", None, None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="data/eval_results/chat_probe.json")
    ap.add_argument("--only", default=None, help="comma-separated question ids")
    a = ap.parse_args()

    sessions: dict = {}
    rows = []
    only = set(a.only.split(",")) if a.only else None
    for qid, question, filters, follow_of in QUESTIONS:
        if only and qid not in only:
            continue
        body = {"question": question, **(filters or {})}
        if follow_of and follow_of in sessions:
            body["session_id"] = sessions[follow_of]
        t0 = time.time()
        try:
            r = requests.post(f"{a.base}/chat", json=body, timeout=400)
            turn = r.json() if r.ok else {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as exc:
            turn = {"error": repr(exc)}
        if "session_id" in turn:
            sessions[qid] = turn["session_id"]
        rows.append({"id": qid, "question": question, "filters": filters,
                     "seconds": round(time.time() - t0, 1), "turn": turn})
        print(f"[{qid}] {rows[-1]['seconds']}s  {str(turn.get('answer', turn))[:90]!r}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
