"""Tests for liq.metrics.qa module."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from liq.metrics.qa import QAResultLike, summarize_qa


@dataclass
class QAResult:
    """Test dataclass matching QAResultLike protocol."""

    missing_ratio: float
    zero_volume_ratio: float
    ohlc_inconsistencies: int
    extreme_moves: int
    negative_volume: int
    non_monotonic_ts: int


class TestSummarizeQA:
    """Tests for summarize_qa function."""

    def test_summarize_qa_from_dataclass(self) -> None:
        """Test summarize_qa with a dataclass input."""
        result = QAResult(
            missing_ratio=0.1,
            zero_volume_ratio=0.05,
            ohlc_inconsistencies=3,
            extreme_moves=2,
            negative_volume=1,
            non_monotonic_ts=0,
        )
        summary = summarize_qa(result)

        assert summary["missing_ratio"] == 0.1
        assert summary["zero_volume_ratio"] == 0.05
        assert summary["ohlc_inconsistencies"] == 3
        assert summary["extreme_moves"] == 2
        assert summary["negative_volume"] == 1
        assert summary["non_monotonic_ts"] == 0

    def test_summarize_qa_from_simple_namespace(self) -> None:
        """Test summarize_qa with a SimpleNamespace input."""
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

    def test_summarize_qa_from_dict(self) -> None:
        """Test summarize_qa with a dict input."""
        data = {
            "missing_ratio": 0.2,
            "zero_volume_ratio": 0.1,
            "ohlc_inconsistencies": 5,
            "extreme_moves": 3,
            "negative_volume": 2,
            "non_monotonic_ts": 1,
        }
        summary = summarize_qa(data)

        assert summary == data

    def test_summarize_qa_none_returns_empty_dict(self) -> None:
        """Test summarize_qa with None returns empty dict."""
        summary = summarize_qa(None)
        assert summary == {}

    def test_summarize_qa_invalid_type_raises_type_error(self) -> None:
        """Test summarize_qa raises TypeError for unsupported types."""
        with pytest.raises(TypeError, match="must be a dict, dataclass, or object"):
            summarize_qa("invalid")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="must be a dict, dataclass, or object"):
            summarize_qa(123)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="must be a dict, dataclass, or object"):
            summarize_qa([1, 2, 3])  # type: ignore[arg-type]

    def test_summarize_qa_incomplete_namespace_raises_type_error(self) -> None:
        """Test summarize_qa raises TypeError for objects missing required attributes."""
        incomplete = SimpleNamespace(missing_ratio=0.1)  # Missing other fields

        with pytest.raises(TypeError, match="must be a dict, dataclass, or object"):
            summarize_qa(incomplete)  # type: ignore[arg-type]

    def test_qa_result_like_protocol(self) -> None:
        """Test QAResultLike protocol is runtime checkable."""
        result = QAResult(0.1, 0.0, 1, 2, 0, 0)
        assert isinstance(result, QAResultLike)

        ns = SimpleNamespace(
            missing_ratio=0.1,
            zero_volume_ratio=0.0,
            ohlc_inconsistencies=1,
            extreme_moves=2,
            negative_volume=0,
            non_monotonic_ts=0,
        )
        assert isinstance(ns, QAResultLike)

    def test_summarize_qa_zero_values(self) -> None:
        """Test summarize_qa with all zero values."""
        result = QAResult(0.0, 0.0, 0, 0, 0, 0)
        summary = summarize_qa(result)

        assert all(v == 0 or v == 0.0 for v in summary.values())

    def test_summarize_qa_preserves_dict_extra_keys(self) -> None:
        """Test summarize_qa preserves extra keys in dict input."""
        data = {
            "missing_ratio": 0.1,
            "extra_key": "extra_value",
        }
        summary = summarize_qa(data)

        assert summary["extra_key"] == "extra_value"
