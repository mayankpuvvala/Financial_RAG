"""
Build a numeric test set whose ground truth is the figure the company
reported to the SEC (XBRL), restricted to filings actually indexed here.

    python -m evaluation.build_test_set            # writes data/test_sets/numeric_xbrl.json

Stratified and seeded so every run builds the same set:
  single    N per metric (revenue / net_income / operating_income / rd_expense)
  multi     two companies, same metric & year
  temporal  one company, one metric, three consecutive years
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import COMPANIES, settings  # noqa: E402
from facts.xbrl import METRIC_LABEL, get_figure  # noqa: E402

BUNDLED = {c["ticker"]: c["name"] for c in COMPANIES}
SHORT = {  # how a person would actually say it
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon",
    "JPM": "JPMorgan Chase", "WFC": "Wells Fargo", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "BLK": "BlackRock", "STT": "State Street", "TROW": "T. Rowe Price", "IVZ": "Invesco",
}
METRICS = ["revenue", "net_income", "operating_income", "rd_expense"]
ASK = {
    "revenue":          ["What was {c}'s total revenue in fiscal year {y}?", "How much revenue did {c} report for FY{y}?"],
    "net_income":       ["What was {c}'s net income in fiscal year {y}?", "How much profit did {c} earn in FY{y}?"],
    "operating_income": ["What was {c}'s operating income in FY{y}?", "How much operating income did {c} report for fiscal {y}?"],
    "rd_expense":       ["How much did {c} spend on research and development in fiscal {y}?",
                         "What were {c}'s R&D expenses in FY{y}?"],
}


def indexed_years() -> dict:
    root = Path(settings.qdrant_path) / "collection"
    out: dict = {}
    for d in root.iterdir():
        ticker, _, year = d.name.rpartition("_")
        if year.isdigit() and ticker in BUNDLED:
            out.setdefault(ticker, set()).add(int(year))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-metric", type=int, default=10)
    ap.add_argument("--multi", type=int, default=8)
    ap.add_argument("--temporal", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=settings.test_sets_dir / "numeric_xbrl.json")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    years = indexed_years()
    facts = {}   # (ticker, metric, year) -> Figure
    for t, ys in years.items():
        for m in METRICS:
            for y in ys:
                f = get_figure(t, m, y)
                if f:
                    facts[(t, m, y)] = f

    cases = []

    def expected(*figs):
        return [{"label": f"{f.ticker} FY{f.fiscal_year} {f.metric}", "value_usd": f.value} for f in figs]

    for m in METRICS:
        pool = sorted(k for k in facts if k[1] == m)
        for t, _, y in rng.sample(pool, min(args.per_metric, len(pool))):
            q = rng.choice(ASK[m]).format(c=SHORT[t], y=y)
            cases.append({"id": f"single-{t}-{m}-{y}", "group": "single", "metric": m, "question": q,
                          "expected": expected(facts[(t, m, y)])})

    multi_pool = sorted({(m, y) for (_, m, y) in facts})
    made = 0
    rng.shuffle(multi_pool)
    for m, y in multi_pool:
        ts = sorted(t for (t, mm, yy) in facts if mm == m and yy == y)
        if len(ts) < 2 or made >= args.multi:
            continue
        a, b = rng.sample(ts, 2)
        q = f"Compare {SHORT[a]} and {SHORT[b]} {METRIC_LABEL[m].split(' (')[0]} in fiscal {y}."
        cases.append({"id": f"multi-{a}-{b}-{m}-{y}", "group": "multi", "metric": m, "question": q,
                      "expected": expected(facts[(a, m, y)], facts[(b, m, y)])})
        made += 1

    trend_pool = sorted({(t, m) for (t, m, _) in facts
                         if all((t, m, y) in facts for y in (2023, 2024, 2025))})
    rng.shuffle(trend_pool)
    for t, m in trend_pool[: args.temporal]:
        q = f"How did {SHORT[t]}'s {METRIC_LABEL[m].split(' (')[0]} change from 2023 to 2025?"
        cases.append({"id": f"temporal-{t}-{m}", "group": "temporal", "metric": m, "question": q,
                      "expected": expected(*(facts[(t, m, y)] for y in (2023, 2024, 2025)))})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    by_group = {}
    for c in cases:
        by_group[c["group"]] = by_group.get(c["group"], 0) + 1
    print(f"wrote {len(cases)} cases -> {args.out}  {by_group}")


if __name__ == "__main__":
    main()
