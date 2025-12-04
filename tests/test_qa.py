from types import SimpleNamespace

from liq.metrics.qa import summarize_qa


def test_summarize_qa_from_dataclass_like() -> None:
    ns = SimpleNamespace(
        missing_ratio=0.1,
        zero_volume_ratio=0.0,
        ohlc_inconsistencies=1,
        extreme_moves=2,
        negative_volume=0,
        non_monotonic_ts=0,
    )
    summary = summarize_qa(ns)
    assert summary["missing_ratio"] == 0.1
    assert summary["ohlc_inconsistencies"] == 1
