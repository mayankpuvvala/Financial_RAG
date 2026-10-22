"""
Regression check for everything that is NOT a headline XBRL figure:
business / risk / legal / cybersecurity descriptions, derived numbers,
and the early-exit paths (out of scope, unknown company, no company).

Each case passes when at least `min_hits` of its regexes match the answer
(case-insensitive). No LLM judge — keyword evidence is deliberately coarse,
so it only catches real regressions (an empty, refused or off-topic answer).

    python -m evaluation.qualitative_eval --systems baseline,agent --trials 1
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger  # noqa: E402

from config import settings  # noqa: E402

CASES = [
    {"id": "apple-products", "q": "What does Apple sell?",
     "any": [r"iphone", r"\bmac\b", r"ipad", r"wearables|watch|airpods", r"services"], "min": 3},
    {"id": "msft-segments", "q": "What are Microsoft's main business segments?",
     "any": [r"productivity and business processes", r"intelligent cloud", r"more personal computing"], "min": 2},
    {"id": "amzn-risks", "q": "What are the key risk factors Amazon discloses in its 10-K?",
     "any": [r"competit", r"supply|inventory|fulfil", r"regulat|legal|tax", r"cyber|security|data", r"international|foreign|currency"], "min": 3},
    {"id": "googl-legal", "q": "Summarize Alphabet's legal proceedings.",
     "any": [r"antitrust", r"department of justice|doj", r"european commission|european union|\beu\b", r"epic|privacy|patent"], "min": 2},
    {"id": "jpm-cyber", "q": "How does JPMorgan describe its cybersecurity risk management?",
     "any": [r"chief information security|ciso", r"cyber", r"board|committee", r"third.party|incident"], "min": 3},
    {"id": "gs-business", "q": "What does Goldman Sachs do?",
     "any": [r"investment bank", r"asset (and|&) wealth|asset management", r"global markets|trading", r"platform solutions|consumer"], "min": 2},
    {"id": "aapl-op-margin", "q": "What was Apple's operating margin in fiscal 2024?",
     "any": [r"31\.[45]\d?\s?%"], "min": 1},
    {"id": "aapl-cash", "q": "How much cash and cash equivalents did Apple have at the end of fiscal 2024?",
     "any": [r"29[,\s ]?943|29\.9\s?billion"], "min": 1},
    {"id": "msft-tax", "q": "What was Microsoft's effective tax rate in fiscal 2024?",
     "any": [r"\b18\s?%"], "min": 1},
    {"id": "out-of-scope", "q": "What is the price of Bitcoin today?",
     "any": [r"outside the scope"], "min": 1},
    {"id": "private-co", "q": "What was SpaceX's revenue in 2024?",
     "any": [r"couldn.t find|no sec filings|doesn.t appear|not a public|aren.t available"], "min": 1},
    {"id": "no-company", "q": "What was the revenue in 2024?",
     "any": [r"which company|outside the scope"], "min": 1},   # classifier treats "no company named" as out of scope
]


def score(answer: str, case: dict) -> dict:
    hits = [p for p in case["any"] if re.search(p, answer, re.IGNORECASE)]
    return {"passed": len(hits) >= case["min"], "hits": len(hits)}


def _system(name):
    from query import ask, ask_baseline
    return {"agent": ask, "baseline": ask_baseline}[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="baseline,agent")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--out", default=str(settings.data_dir / "eval_results" / "qualitative.jsonl"))
    a = ap.parse_args()
    logger.remove(); logger.add(sys.stderr, level="ERROR")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)

    for trial in range(a.trials):
        for name in a.systems.split(","):
            ask = _system(name)
            for case in CASES:
                t0 = time.time()
                try:
                    r = ask(case["q"]); answer, ver = r.answer, r.verification or {}
                    err = None
                except Exception as exc:
                    answer, ver, err = "", {}, repr(exc)
                s = score(answer, case)
                row = {"system": name, "trial": trial, "id": case["id"], "passed": s["passed"],
                       "hits": s["hits"], "min": case["min"], "seconds": round(time.time() - t0, 1),
                       "verify": ver.get("status"), "steps": ver.get("steps"),
                       "unverified_note": "could not be matched" in answer, "error": err, "answer": answer[:1500]}
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{name} t{trial}] {'PASS' if s['passed'] else 'FAIL'} {case['id']} "
                      f"({s['hits']}/{case['min']}) {row['seconds']}s verify={row['verify']}", flush=True)

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    print("\nSummary")
    for name in a.systems.split(","):
        rs = [r for r in rows if r["system"] == name]
        if rs:
            print(f"  {name:9} {sum(r['passed'] for r in rs)}/{len(rs)} passed, "
                  f"{sum(r['unverified_note'] for r in rs)} carried an 'unverified figures' note")


if __name__ == "__main__":
    main()
