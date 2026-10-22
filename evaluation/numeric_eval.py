"""
Deterministic accuracy eval — no LLM judge.

A case passes when EVERY expected figure appears in the answer (matched with
the verifier's rounding-aware comparison). Runs the original linear pipeline
("baseline") and the LangGraph agent ("agent") on the same questions.

    python -m evaluation.numeric_eval --systems baseline,agent --trials 1
    python -m evaluation.numeric_eval --systems agent --ids single-AAPL-revenue-2024 --trials 5

Results are appended to a JSONL file as they finish, so a long run can be
interrupted and resumed (finished (system, trial, id) triples are skipped).
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger  # noqa: E402

from config import settings  # noqa: E402
from verification.verify import amount_matches, extract_amounts  # noqa: E402

DEFAULT_SET = settings.test_sets_dir / "numeric_xbrl.json"
DEFAULT_OUT = settings.data_dir / "eval_results" / "numeric_eval.jsonl"


def score(answer: str, expected: list) -> dict:
    amounts = extract_amounts(answer)
    found = {e["label"]: any(amount_matches(a, e["value_usd"]) for a in amounts) for e in expected}
    return {"passed": all(found.values()), "found": found}


def _system(name: str):
    if name == "baseline":
        from query import ask_baseline
        return ask_baseline
    if name == "agent":
        from query import ask
        return ask
    raise SystemExit(f"unknown system {name!r}")


def _done(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            out.add((r["system"], r["trial"], r["id"]))
        except Exception:
            pass
    return out


def run(args) -> None:
    cases = json.loads(Path(args.test_set).read_text(encoding="utf-8"))
    if args.ids:
        wanted = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done(out)
    systems = args.systems.split(",")

    for trial in range(args.trials):
        for name in systems:
            ask = _system(name)
            for i, case in enumerate(cases, 1):
                key = (name, trial, case["id"])
                if key in done:
                    continue
                t0 = time.time()
                try:
                    r = ask(case["question"])
                    answer, steps, vstatus = r.answer, (r.verification or {}).get("steps"), (r.verification or {}).get("status")
                    error = None
                except Exception as exc:    # a crash is a failed answer, not a skipped one
                    answer, steps, vstatus, error = "", None, None, repr(exc)
                s = score(answer, case["expected"])
                row = {"system": name, "trial": trial, "id": case["id"], "group": case["group"],
                       "metric": case["metric"], "passed": s["passed"], "found": s["found"],
                       "seconds": round(time.time() - t0, 1), "verify": vstatus, "steps": steps,
                       "error": error, "answer": answer[:1200]}
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{name} t{trial} {i}/{len(cases)}] {'PASS' if s['passed'] else 'FAIL'} "
                      f"{case['id']} ({row['seconds']}s)", flush=True)
                time.sleep(args.delay)
    summarize(out)


def summarize(path: Path) -> None:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return

    def table(title, keyf):
        agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))
        for r in rows:
            cell = agg[keyf(r)][r["system"]]
            cell[0] += r["passed"]; cell[1] += 1; cell[2] += r["seconds"]
        systems = sorted({r["system"] for r in rows})
        print(f"\n{title}")
        print(f"{'':<22}" + "".join(f"{s:>26}" for s in systems))
        for k in sorted(agg):
            line = f"{str(k):<22}"
            for s in systems:
                p, n, sec = agg[k][s]
                line += f"{(f'{p}/{n} = {100*p/n:.0f}%  {sec/n:.0f}s/q' if n else '-'):>26}"
            print(line)

    table("By group", lambda r: r["group"])
    table("By metric", lambda r: r["metric"])
    table("Overall", lambda r: "ALL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", default=str(DEFAULT_SET))
    ap.add_argument("--systems", default="baseline,agent")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--ids", default=None, help="comma-separated case ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    if a.summary_only:
        summarize(Path(a.out))
    else:
        run(a)
