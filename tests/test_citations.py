from generation.synthesizer import renumber_citations


def _cits(*idx):
    return [{"index": i, "ticker": "X"} for i in idx]


def test_no_chained_rewrite():
    # Old str.replace chain: [1]->[3] then [3]->[5] turned every marker into [5].
    text, cits = renumber_citations("A [1], B [2], C [3].", _cits(1, 2, 3), 2)
    assert text == "A [3], B [4], C [5]."
    assert [c["index"] for c in cits] == [3, 4, 5]


def test_zero_offset_is_identity():
    text, cits = renumber_citations("A [1] B [2]", _cits(1, 2), 0)
    assert text == "A [1] B [2]"
    assert [c["index"] for c in cits] == [1, 2]


def test_unknown_markers_untouched():
    text, _ = renumber_citations("Value [9] and [1]", _cits(1), 4)
    assert text == "Value [9] and [5]"


def test_adjacent_markers():
    text, _ = renumber_citations("x[1][2]", _cits(1, 2), 3)
    assert text == "x[4][5]"
