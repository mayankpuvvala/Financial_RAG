import facts.xbrl as X


def row(end, val, filed, start=None, unit="USD", accn="a"):
    return {"u": unit, "s": start, "e": end, "v": val, "a": accn, "d": filed, "f": "10-K"}


def fake_load(monkeypatch, concept_rows):
    monkeypatch.setattr(X, "_load", lambda t: concept_rows)


def test_original_filing_wins_over_later_recast(monkeypatch):
    fake_load(monkeypatch, {"NetIncomeLoss": [
        row("2024-12-31", 26_973, "2026-02-20", start="2024-01-01", accn="fy25"),   # recast
        row("2024-12-31", 27_132, "2025-02-25", start="2024-01-01", accn="fy24"),   # as filed
    ]})
    f = X.get_figure("BAC", "net_income", 2024)
    assert f.value == 27_132 and f.accession == "fy24"


def test_only_full_year_durations_count(monkeypatch):
    fake_load(monkeypatch, {"Revenues": [
        row("2024-12-31", 10, "2025-02-01", start="2024-10-01"),    # a quarter
        row("2024-12-31", 40, "2025-02-01", start="2024-01-01"),    # the year
    ]})
    assert X.get_figure("X", "revenue", 2024).value == 40


def test_revenue_takes_largest_concept(monkeypatch):
    fake_load(monkeypatch, {
        "RevenueFromContractWithCustomerExcludingAssessedTax": [row("2024-12-31", 20, "2025-02-01", start="2024-01-01")],
        "Revenues": [row("2024-12-31", 50, "2025-02-01", start="2024-01-01")],
    })
    f = X.get_figure("X", "revenue", 2024)
    assert f.value == 50 and f.concept == "Revenues"


def test_instant_metric_has_no_start(monkeypatch):
    fake_load(monkeypatch, {"Assets": [row("2024-09-28", 365, "2024-11-01")]})
    assert X.get_figure("X", "total_assets", 2024).value == 365


def test_missing_returns_none(monkeypatch):
    fake_load(monkeypatch, {})
    assert X.get_figure("X", "revenue", 2024) is None
    assert X.get_figure("X", "not_a_metric", 2024) is None


def test_early_january_year_end_belongs_to_prior_fiscal_year():
    assert X.fiscal_year_of("2025-01-03") == 2024
    assert X.fiscal_year_of("2025-01-26") == 2025   # NVIDIA-style late-January year end
    assert X.fiscal_year_of("2024-09-28") == 2024
