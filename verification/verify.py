"""
Answer verifier — checks the dollar figures in a generated answer.

Two independent checks:

  1. Key figure   — for headline metrics (revenue, R&D, net income,
                    operating income) the answer must contain the value SEC
                    published as XBRL for that ticker/year. A different
                    number in its place is a hard "mismatch".
  2. Support      — every other explicit-unit dollar amount must appear in
                    the retrieved context (under any of $, $K, $M scale), or
                    be derivable from other figures in the answer (a
                    difference or a sum). Otherwise it's "unsupported",
                    i.e. likely invented.

Percentages and share counts are not checked.
"""

import bisect
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from facts.xbrl import Figure, FOCUS_TO_METRIC, get_figure

_UNITS = {
    "trillion": 1e12, "tn": 1e12,
    "billion": 1e9,   "bn": 1e9,
    "million": 1e6,   "mm": 1e6,
    "thousand": 1e3,
}

_AMOUNT = re.compile(
    r"(?P<dollar>\$\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<unit>trillion|billion|million|thousand|tn|bn|mm)\b)?",
    re.IGNORECASE,
)
_BARE_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{4,}(?:\.\d+)?")

REL_TOL     = 0.005   # a stated figure may differ from the target by 0.5%
DERIVED_TOL = 0.015   # differences/sums of rounded figures drift a little more
MIN_USD     = 1e7     # ignore tiny amounts (per-share values, counts, ...)


@dataclass
class Amount:
    raw:        str
    candidates: List[float]   # possible dollar values (several when the unit is implicit)
    half_unit:  float          # rounding slack implied by the digits written
    explicit:   bool           # carried an explicit unit word


def _half_unit(num: str, scale: float) -> float:
    decimals = len(num.split(".")[1]) if "." in num else 0
    return 0.5 * (10 ** -decimals) * scale


# gpt-oss formats "391,035" as "391 035" (narrow no-break space) at times.
_DIGIT_GROUP_SPACE = re.compile(r"(?<=\d)[    ](?=\d{3}(?!\d))")
_ODD_SPACES        = re.compile(r"[    ]")


def normalize(text: str) -> str:
    text = _DIGIT_GROUP_SPACE.sub("", text)
    return _ODD_SPACES.sub(" ", text).replace("−", "-")


def extract_amounts(text: str) -> List[Amount]:
    """Dollar amounts stated in `text` (needs a $ sign or a unit word)."""
    text = normalize(text)
    out: List[Amount] = []
    for m in _AMOUNT.finditer(text):
        # "[1]" style citation markers are not amounts
        s, e = m.span()
        if s > 0 and text[s - 1] == "[" and e < len(text) and text[e] == "]":
            continue
        unit = (m.group("unit") or "").lower()
        if not unit and not m.group("dollar"):
            continue
        num = m.group("num")
        value = float(num.replace(",", ""))
        if unit:
            scale = _UNITS[unit]
            out.append(Amount(m.group(0).strip(), [value * scale], _half_unit(num, scale), True))
        else:
            # "$391,035" — unit is implicit; filings quote millions or thousands.
            out.append(Amount(
                m.group(0).strip(),
                [value, value * 1e3, value * 1e6],
                _half_unit(num, 1.0),
                False,
            ))
    return out


def _close(a: float, target: float, half_unit: float = 0.0, rel: float = REL_TOL) -> bool:
    return abs(a - target) <= max(half_unit, rel * abs(target))


def amount_matches(amount: Amount, target: float) -> bool:
    t = abs(target)
    return any(_close(abs(c), t, amount.half_unit) for c in amount.candidates)


@dataclass
class Verdict:
    status:      str                                   # verified | mismatch | unsupported | unchecked
    details:     str = ""
    targets:     List[dict] = field(default_factory=list)   # XBRL figures checked
    missing:     List[str] = field(default_factory=list)    # targets absent from the answer
    unsupported: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("verified", "unchecked")

    def to_dict(self) -> dict:
        return {"status": self.status, "details": self.details, "targets": self.targets,
                "missing": self.missing, "unsupported": self.unsupported}


def _context_values(texts: Iterable[str]) -> List[float]:
    """Every dollar value the context could be saying, at $, $K and $M scale."""
    vals: List[float] = []
    for text in texts:
        for a in extract_amounts(text):
            if a.explicit:
                vals.extend(a.candidates)
        for m in _BARE_NUM.finditer(text):
            tok = m.group(0)
            v = float(tok.replace(",", ""))
            if "," not in tok and 1900 <= v <= 2100:   # a year, not an amount
                continue
            vals.extend((v, v * 1e3, v * 1e6))
    return sorted(set(vals))


def _in_sorted(vals: Sequence[float], target: float, half_unit: float) -> bool:
    tol = max(half_unit, REL_TOL * abs(target))
    i = bisect.bisect_left(vals, target - tol)
    return i < len(vals) and vals[i] <= target + tol


def _derivable(a: Amount, others: List[float]) -> bool:
    """Is `a` a sum or difference of two other figures stated in the answer?"""
    for target in a.candidates:
        for i, x in enumerate(others):
            for y in others[i + 1:]:
                if _close(target, abs(x - y), a.half_unit, DERIVED_TOL) or \
                   _close(target, x + y, a.half_unit, DERIVED_TOL):
                    return True
    return False


def targets_for(tickers: Sequence[str], years: Sequence[int], focus: str) -> List[Figure]:
    metric = FOCUS_TO_METRIC.get(focus)
    if not metric:
        return []
    figs = []
    for t in tickers:
        for y in years:
            fig = get_figure(t, metric, y)
            if fig:
                figs.append(fig)
    return figs


def verify_answer(
    answer:        str,
    context_texts: Sequence[str],
    figures:       Sequence[Figure] = (),
    require_all:   bool = False,
) -> Verdict:
    """
    `figures`     — XBRL values this answer is expected to state.
    `require_all` — every figure must appear (a question pinned to specific
                    year(s)); otherwise one matching figure is enough, since
                    a question without a year may legitimately answer with
                    just the latest.
    """
    amounts = extract_amounts(answer)
    targets = [f.to_dict() for f in figures]

    # --- 1. key figure --------------------------------------------------
    if figures:
        hit  = [f for f in figures if any(amount_matches(a, f.value) for a in amounts)]
        miss = [f for f in figures if f not in hit]
        failed = (len(hit) == 0) or (require_all and miss)
        if failed:
            expected = "; ".join(f"{f.ticker} FY{f.fiscal_year} {f.metric} = {f.human()}" for f in miss or figures)
            return Verdict(
                "mismatch",
                f"answer does not state the SEC-reported figure(s): {expected}",
                targets=targets,
                missing=[f"{f.ticker} FY{f.fiscal_year}" for f in miss],
            )

    # --- 2. support for the remaining amounts -----------------------------
    ctx_vals = _context_values(context_texts)
    fig_vals = [abs(f.value) for f in figures if f.unit == "USD"]
    explicit = [a for a in amounts if a.explicit and max(a.candidates) >= MIN_USD]
    stated   = [c for a in explicit for c in a.candidates]
    unsupported: List[str] = []
    for a in explicit:
        v = a.candidates[0]
        if any(_close(v, fv, a.half_unit) for fv in fig_vals):
            continue
        if any(_in_sorted(ctx_vals, c, a.half_unit) for c in a.candidates):
            continue
        others = [x for x in stated if x != v]
        if _derivable(a, others):
            continue
        unsupported.append(a.raw)

    if unsupported:
        return Verdict(
            "unsupported",
            "figure(s) not found in the retrieved sources: " + ", ".join(unsupported),
            targets=targets, unsupported=unsupported,
        )

    return Verdict("verified" if figures else "unchecked", targets=targets)


# Catches the model's own "not found" phrasing. Deliberately over-inclusive
# (a false positive costs one extra, usually-redundant retry; a false
# negative silently ships a wrong "not found").
_REFUSAL_PATTERN = re.compile(
    r"cannot be (?:found|determined|answered)|can.t be (?:found|determined)|"
    r"does not (?:explicitly )?(?:mention|state|provide|discuss|contain|specify|disclose)|"
    r"\bnot (?:explicitly )?(?:stated|mentioned|found|available|provided|disclosed|specified|included)\b|"
    r"no (?:relevant )?information",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_PATTERN.search(answer))
