from facts.xbrl import Figure
from verification.verify import extract_amounts, amount_matches, verify_answer


def fig(value, ticker="AAPL", metric="revenue", fy=2024):
    return Figure(ticker=ticker, metric=metric, fiscal_year=fy, value=value, unit="USD",
                  concept="X", period_start="2023-10-01", period_end="2024-09-28",
                  form="10-K", accession="0000-00-000000", filed="2024-11-01")


AAPL_REV = fig(391_035_000_000)


def test_extract_ignores_citation_markers_and_years():
    amts = extract_amounts("In 2024, revenue was $391,035 million [1] (~$391 billion).")
    assert len(amts) == 2
    assert all(amount_matches(a, 391_035e6) for a in amts)


def test_rounded_forms_match():
    for text in ("$391,035 million", "$391.0 billion", "about $391 billion", "391,035 million"):
        assert any(amount_matches(a, 391_035e6) for a in extract_amounts(text)), text


def test_wrong_year_value_is_not_a_match():
    # Apple FY2023 revenue, 383,285 — must not pass for FY2024.
    assert not any(amount_matches(a, 391_035e6) for a in extract_amounts("$383,285 million"))


def test_unitless_dollar_amount_read_as_millions():
    assert any(amount_matches(a, 391_035e6) for a in extract_amounts("$391,035"))


def test_verified_when_figure_present():
    v = verify_answer("Apple's FY2024 net sales were $391,035 million [1].", ["Total net sales | 391035"], [AAPL_REV])
    assert v.status == "verified" and v.ok


def test_mismatch_when_wrong_number():
    v = verify_answer("Apple's FY2024 net sales were $383,285 million [1].", ["383285 391035"], [AAPL_REV])
    assert v.status == "mismatch" and not v.ok


def test_mismatch_when_no_number_at_all():
    v = verify_answer("Not found in the provided context.", [], [AAPL_REV])
    assert v.status == "mismatch"


def test_require_all_years():
    f23 = fig(383_285_000_000, fy=2023)
    ans = "FY2024 revenue was $391,035 million."
    assert verify_answer(ans, ["391035 383285"], [AAPL_REV, f23]).status == "verified"
    v = verify_answer(ans, ["391035 383285"], [AAPL_REV, f23], require_all=True)
    assert v.status == "mismatch" and v.missing == ["AAPL FY2023"]


def test_unsupported_number_flagged():
    v = verify_answer("Revenue was $391,035 million and R&D was $99.9 billion.",
                      ["Total net sales 391035"], [AAPL_REV])
    assert v.status == "unsupported" and "$99.9 billion" in v.unsupported[0]


def test_context_in_thousands_supports_answer():
    # Table quoted in thousands: 45183036 -> $45.2 billion
    v = verify_answer("Revenue was $45.2 billion.", ["Revenues | 45183036"], [])
    assert v.status == "unchecked"


def test_derived_difference_is_not_flagged():
    ans = "Alphabet spent $49.3 billion vs Microsoft $29.5 billion, a gap of $19.8 billion."
    v = verify_answer(ans, ["research and development 49326 29510"], [])
    assert v.ok, v.details


def test_loss_matches_by_magnitude():
    f = fig(-2_500_000_000, metric="net_income")
    assert verify_answer("The company reported a net loss of $2.5 billion.", [], [f]).status == "verified"


def test_narrow_nbsp_thousands_separator():
    # Seen in real gpt-oss output: "$391 035 million"
    amts = extract_amounts("Net sales were $391 035 million.")
    assert any(amount_matches(a, 391_035e6) for a in amts)


def test_refusal_phrasings():
    from verification.verify import is_refusal
    for t in ("That is not stated in the provided context.",
              "The 10-K does not mention this.",
              "No relevant information was found.",
              "The figure cannot be determined from the sources."):
        assert is_refusal(t), t
    assert not is_refusal("Apple's revenue was $391,035 million [1].")
