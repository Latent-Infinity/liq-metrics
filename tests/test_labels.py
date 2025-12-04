from liq.metrics.labels import summarize_labels


def test_summarize_labels_counts() -> None:
    labels = [1, -1, 0, 1, 0]
    summary = summarize_labels(labels)
    assert summary["positive"] == 2
    assert summary["negative"] == 1
    assert summary["neutral"] == 2
