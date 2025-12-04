"""Tests for liq.metrics.labels module."""

import pytest

from liq.metrics.labels import summarize_labels


class TestSummarizeLabels:
    """Tests for summarize_labels function."""

    def test_summarize_labels_basic(self) -> None:
        """Test basic label counting."""
        labels = [1, -1, 0, 1, 0]
        summary = summarize_labels(labels)

        assert summary["positive"] == 2
        assert summary["negative"] == 1
        assert summary["neutral"] == 2

    def test_summarize_labels_empty_returns_zeros(self) -> None:
        """Test empty input returns zero counts."""
        summary = summarize_labels([])

        assert summary["positive"] == 0
        assert summary["negative"] == 0
        assert summary["neutral"] == 0

    def test_summarize_labels_all_positive(self) -> None:
        """Test with all positive labels."""
        summary = summarize_labels([1, 2, 3, 100])

        assert summary["positive"] == 4
        assert summary["negative"] == 0
        assert summary["neutral"] == 0

    def test_summarize_labels_all_negative(self) -> None:
        """Test with all negative labels."""
        summary = summarize_labels([-1, -2, -3, -100])

        assert summary["positive"] == 0
        assert summary["negative"] == 4
        assert summary["neutral"] == 0

    def test_summarize_labels_all_neutral(self) -> None:
        """Test with all neutral labels."""
        summary = summarize_labels([0, 0, 0, 0])

        assert summary["positive"] == 0
        assert summary["negative"] == 0
        assert summary["neutral"] == 4

    def test_summarize_labels_single_positive(self) -> None:
        """Test with single positive label."""
        summary = summarize_labels([1])

        assert summary["positive"] == 1
        assert summary["negative"] == 0
        assert summary["neutral"] == 0

    def test_summarize_labels_single_negative(self) -> None:
        """Test with single negative label."""
        summary = summarize_labels([-1])

        assert summary["positive"] == 0
        assert summary["negative"] == 1
        assert summary["neutral"] == 0

    def test_summarize_labels_single_neutral(self) -> None:
        """Test with single neutral label."""
        summary = summarize_labels([0])

        assert summary["positive"] == 0
        assert summary["negative"] == 0
        assert summary["neutral"] == 1

    def test_summarize_labels_generator_input(self) -> None:
        """Test labels accepts generator/iterator input."""
        summary = summarize_labels(x for x in [1, -1, 0])

        assert summary["positive"] == 1
        assert summary["negative"] == 1
        assert summary["neutral"] == 1

    def test_summarize_labels_large_values(self) -> None:
        """Test with large positive/negative values."""
        summary = summarize_labels([1000000, -1000000, 0])

        assert summary["positive"] == 1
        assert summary["negative"] == 1
        assert summary["neutral"] == 1

    def test_summarize_labels_invalid_type_raises_type_error(self) -> None:
        """Test labels raises TypeError for non-integer values."""
        with pytest.raises(TypeError, match="must be integers"):
            summarize_labels([0.5, 1.5])  # type: ignore[list-item]

        with pytest.raises(TypeError, match="must be integers"):
            summarize_labels([1, "invalid", 0])  # type: ignore[list-item]

        with pytest.raises(TypeError, match="must be integers"):
            summarize_labels([None])  # type: ignore[list-item]

    def test_summarize_labels_bool_raises_type_error(self) -> None:
        """Test labels raises TypeError for boolean values (despite being int subclass)."""
        with pytest.raises(TypeError, match="must be integers"):
            summarize_labels([True, False])  # type: ignore[list-item]

    def test_summarize_labels_returns_dict_with_correct_keys(self) -> None:
        """Test that returned dict always has all three keys."""
        summary = summarize_labels([])

        assert "positive" in summary
        assert "negative" in summary
        assert "neutral" in summary
        assert len(summary) == 3

    def test_summarize_labels_values_are_integers(self) -> None:
        """Test that returned values are integers."""
        summary = summarize_labels([1, -1, 0])

        assert isinstance(summary["positive"], int)
        assert isinstance(summary["negative"], int)
        assert isinstance(summary["neutral"], int)

    def test_summarize_labels_many_items(self) -> None:
        """Test with many labels."""
        labels = [1] * 1000 + [-1] * 500 + [0] * 250
        summary = summarize_labels(labels)

        assert summary["positive"] == 1000
        assert summary["negative"] == 500
        assert summary["neutral"] == 250
