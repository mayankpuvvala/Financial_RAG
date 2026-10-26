"""
Review a deployed API's chat history: mark verifiable turns correct/incorrect
(figures checked against SEC XBRL) and delete sessions with no content.

Dry run by default; nothing is written without --apply. A dated JSON backup of every
session is written first (prod_sessions_snapshot_<date>.json, gitignored). Turns that
already have a review are never overwritten.

    python -m evaluation.review_history --base https://<app>.up.railway.app
    python -m evaluation.review_history --base https://<app>.up.railway.app --apply
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from facts.xbrl import get_figure  # noqa: E402
from verification.verify import amount_matches, extract_amounts  # noqa: E402

# question regex -> (ticker, metric, fiscal_year): the answer must state the SEC figure.
NUMERIC_RULES = [
    (r"apple.*(total revenue|net sales).*fy ?2024|apple.*(total revenue|net sales).*fiscal year 2024", ("AAPL", "revenue", 2024)),
    (r"microsoft.*research and development.*fy ?2024", ("MSFT", "rd_expense", 2024)),
    (r"goldman sachs.*net income.*fiscal year 2024", ("GS", "net_income", 2024)),
    (r"profit of meta in 2025", ("META", "net_income", 2025)),
    (r"pnc financial services.*profit in fy ?24", ("PNC", "net_income", 2024)),
    (r"total revenue reported by apple inc\. in its most recent fiscal year", ("AAPL", "revenue", 2025)),
    (r"total revenue.*by the jpm company for the most recent fiscal year", ("JPM", "revenue", 2025)),
]
# Answerable-from-a-10-K questions that the system wrongly refused or failed.
SHOULD_ANSWER = [
    r"blackrock's balance sheet show for total assets",
    r"cash and cash equivalents.*trow",
]
SYSTEM_FAILURE = re.compile(r"usage limit|temporarily unavailable|Which company are you asking about", re.I)
# Sessions with nothing worth keeping: greetings, off-topic, or only error replies.
JUNK_TITLES = [r"^hi$", r"indian stocks"]


def verdict(question: str, answer: str):
    q = question.lower()
    for pattern, (ticker, metric, year) in NUMERIC_RULES:
        if re.search(pattern, q):
            fig = get_figure(ticker, metric, year)
            if not fig:
                return None, f"no XBRL figure for {ticker} {metric} {year}"
            ok = any(amount_matches(a, fig.value) for a in extract_amounts(answer))
            return ok, f"{'states' if ok else 'does not state'} {ticker} FY{year} {metric} = {fig.human()}"
    if SYSTEM_FAILURE.search(answer):
        return False, "system failure message instead of an answer"
    if any(re.search(p, q) for p in SHOULD_ANSWER) and re.search(r"cannot be found|does not (contain|mention)|no relevant", answer, re.I):
        return False, "refused a question the filings can answer"
    return None, "not verifiable here — left unreviewed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup", default=f"prod_sessions_snapshot_{__import__('datetime').date.today()}.json")
    a = ap.parse_args()
    base = a.base.rstrip("/")

    sessions = requests.get(f"{base}/chat/sessions", timeout=60).json()
    full = {s["id"]: {"meta": s, "turns": requests.get(f"{base}/chat/sessions/{s['id']}", timeout=60).json()} for s in sessions}
    Path(a.backup).write_text(json.dumps(full, indent=1), encoding="utf-8")
    print(f"backup of {len(full)} sessions -> {a.backup}\n")

    to_delete, marks = [], []
    for sid, s in full.items():
        title, turns = s["meta"]["title"], s["turns"]
        only_failures = turns and all(SYSTEM_FAILURE.search(t["answer"]) for t in turns)
        if any(re.search(p, title.strip().lower()) for p in JUNK_TITLES) or only_failures:
            to_delete.append((sid, title, len(turns)))
            continue
        for t in turns:
            ok, why = verdict(t["question"], t["answer"])
            if ok is not None and t.get("is_correct") is None:   # never overwrite a human review
                marks.append((sid, t["id"], ok, t["question"][:70], why))

    print("SESSIONS TO DELETE:")
    for sid, title, n in to_delete:
        print(f"  {sid[:8]}  {n} turn(s)  {title[:70]!r}")
    print("\nTURNS TO MARK:")
    for sid, tid, ok, q, why in marks:
        print(f"  {'CORRECT  ' if ok else 'INCORRECT'} {q!r}\n             -> {why}")

    if not a.apply:
        print("\n(dry run — pass --apply to write)")
        return
    for sid, tid, ok, *_ in marks:
        requests.patch(f"{base}/chat/sessions/{sid}/turns/{tid}", json={"is_correct": ok}, timeout=60).raise_for_status()
    for sid, *_ in to_delete:
        requests.delete(f"{base}/chat/sessions/{sid}", timeout=60).raise_for_status()
    print(f"\napplied: {len(marks)} marks, {len(to_delete)} sessions deleted")


if __name__ == "__main__":
    main()
