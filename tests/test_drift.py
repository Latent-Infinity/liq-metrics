"""Tests for liq.metrics.drift module."""

import pytest

from liq.metrics.drift import summarize_drift


class TestSummarizeDrift:
    """Tests for summarize_drift function."""

    def test_summarize_drift_basic(self) -> None:
        """Test basic drift statistics calculation."""
        stats = summarize_drift([0.1, 0.2, 0.3])

        assert stats["max"] == 0.3
        assert abs(stats["mean"] - 0.2) < 1e-9

    def test_summarize_drift_empty_returns_zeros(self) -> None:
        """Test empty input returns zero values."""
        stats = summarize_drift([])

        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0

    def test_summarize_drift_single_value(self) -> None:
        """Test single value returns same value for max and mean."""
        stats = summarize_drift([0.5])

        assert stats["max"] == 0.5
        assert stats["mean"] == 0.5

    def test_summarize_drift_negative_values(self) -> None:
        """Test drift with negative values."""
        stats = summarize_drift([-0.3, -0.1, 0.2])

        assert stats["max"] == 0.2
        assert abs(stats["mean"] - (-0.2 / 3)) < 1e-9

    def test_summarize_drift_all_same_values(self) -> None:
        """Test drift with all same values."""
        stats = summarize_drift([0.5, 0.5, 0.5])

        assert stats["max"] == 0.5
        assert stats["mean"] == 0.5

    def test_summarize_drift_accepts_integers(self) -> None:
        """Test drift accepts integer inputs."""
        stats = summarize_drift([1, 2, 3])

        assert stats["max"] == 3.0
        assert stats["mean"] == 2.0

    def test_summarize_drift_mixed_int_float(self) -> None:
        """Test drift with mixed int and float."""
        stats = summarize_drift([1, 0.5, 2])

        assert stats["max"] == 2.0
        assert abs(stats["mean"] - (3.5 / 3)) < 1e-9

    def test_summarize_drift_generator_input(self) -> None:
        """Test drift accepts generator/iterator input."""
        stats = summarize_drift(x * 0.1 for x in range(1, 4))

        assert abs(stats["max"] - 0.3) < 1e-9
        assert abs(stats["mean"] - 0.2) < 1e-9

    def test_summarize_drift_invalid_type_raises_type_error(self) -> None:
        """Test drift raises TypeError for non-numeric values."""
        with pytest.raises(TypeError, match="must be numeric"):
            summarize_drift(["a", "b", "c"])  # type: ignore[list-item]

        with pytest.raises(TypeError, match="must be numeric"):
            summarize_drift([0.1, "invalid", 0.3])  # type: ignore[list-item]

        with pytest.raises(TypeError, match="must be numeric"):
            summarize_drift([None])  # type: ignore[list-item]

    def test_summarize_drift_large_values(self) -> None:
        """Test drift with large values."""
        stats = summarize_drift([1e10, 2e10, 3e10])

        assert stats["max"] == 3e10
        assert abs(stats["mean"] - 2e10) < 1e5

    def test_summarize_drift_small_values(self) -> None:
        """Test drift with very small values."""
        stats = summarize_drift([1e-10, 2e-10, 3e-10])

        assert stats["max"] == 3e-10
        assert abs(stats["mean"] - 2e-10) < 1e-15

    def test_summarize_drift_returns_float_for_max(self) -> None:
        """Test that max is always returned as float."""
        stats = summarize_drift([1, 2, 3])  # Integer input

        assert isinstance(stats["max"], float)
        assert isinstance(stats["mean"], float)
