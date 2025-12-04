from liq.metrics.drift import summarize_drift


def test_summarize_drift() -> None:
    stats = summarize_drift([0.1, 0.2, 0.3])
    assert stats["max"] == 0.3
    assert abs(stats["mean"] - 0.2) < 1e-9
