import pytest

from liq.metrics.prediction import summarize_classification, summarize_regression


def test_summarize_classification() -> None:
    y_true = [0, 1, 1, 2, 2, 2]
    y_pred = [0, 0, 1, 2, 1, 2]
    metrics = summarize_classification(y_true, y_pred)
    assert metrics["count"] == 6.0
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 0.8) / 3, rel=1e-6)


def test_summarize_classification_empty() -> None:
    metrics = summarize_classification([], [])
    assert metrics["count"] == 0.0
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_summarize_regression_with_nll_and_coverage() -> None:
    y_true = [1.0, 2.0]
    y_pred = [1.0, 3.0]
    y_log_var = [0.0, 0.0]
    metrics = summarize_regression(y_true, y_pred, y_log_var)
    assert metrics["count"] == 2.0
    assert metrics["corr"] == pytest.approx(1.0)
    assert metrics["nll"] == pytest.approx(1.1689385, rel=1e-6)
    assert metrics["coverage_1sigma"] == pytest.approx(1.0)
    assert metrics["coverage_2sigma"] == pytest.approx(1.0)


def test_summarize_regression_no_log_var() -> None:
    y_true = [1.0, 2.0]
    y_pred = [1.0, 3.0]
    metrics = summarize_regression(y_true, y_pred)
    assert metrics["nll"] == 0.0


def test_summarize_regression_empty() -> None:
    metrics = summarize_regression([], [])
    assert metrics["count"] == 0.0
    assert metrics["corr"] == 0.0
    assert metrics["nll"] == 0.0


def test_classification_errors() -> None:
    with pytest.raises(ValueError):
        summarize_classification([1], [1, 2])
    with pytest.raises(TypeError):
        summarize_classification([1, 2], [True, 1])


def test_regression_errors() -> None:
    with pytest.raises(ValueError):
        summarize_regression([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        summarize_regression([1.0, 2.0], [1.0, 2.0], [0.0])
    with pytest.raises(TypeError):
        summarize_regression([1.0, True], [1.0, 2.0])
    with pytest.raises(ValueError):
        summarize_regression([1.0, 2.0], [1.0, 2.0], [0.0, 0.0], coverage_sigmas=(0,))
