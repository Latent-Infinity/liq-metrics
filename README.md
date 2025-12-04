# liq-metrics

Performance metrics and evaluation utilities for the LIQ Stack ecosystem.

## Installation

```bash
pip install liq-metrics
```

For development:
```bash
pip install liq-metrics[dev]
```

## Overview

`liq-metrics` provides utilities to summarize and evaluate performance metrics from the LIQ Stack ecosystem:

- **QA ingestion**: Flatten liq-data QA results for reporting
- **Drift summaries**: Compute basic statistics for drift signals from liq-features
- **Label summaries**: Count triple-barrier/meta-label outcomes

## Usage

### QA Result Summarization

Convert QA results (dataclasses, SimpleNamespace, or dicts) to flat dictionaries:

```python
from dataclasses import dataclass
from liq.metrics import summarize_qa

@dataclass
class QAResult:
    missing_ratio: float
    zero_volume_ratio: float
    ohlc_inconsistencies: int
    extreme_moves: int
    negative_volume: int
    non_monotonic_ts: int

result = QAResult(
    missing_ratio=0.05,
    zero_volume_ratio=0.01,
    ohlc_inconsistencies=2,
    extreme_moves=1,
    negative_volume=0,
    non_monotonic_ts=0,
)

summary = summarize_qa(result)
# {'missing_ratio': 0.05, 'zero_volume_ratio': 0.01, 'ohlc_inconsistencies': 2, ...}
```

Also accepts dict input:

```python
summary = summarize_qa({"missing_ratio": 0.1, "custom_field": 42})
# {'missing_ratio': 0.1, 'custom_field': 42}
```

### Drift Statistics

Compute max and mean statistics for drift signals:

```python
from liq.metrics import summarize_drift

# From feature pipeline drift detection
drift_values = [0.12, 0.08, 0.15, 0.03]
stats = summarize_drift(drift_values)
# {'max': 0.15, 'mean': 0.095}

# Empty input returns zeros
stats = summarize_drift([])
# {'max': 0.0, 'mean': 0.0}
```

### Label Counting

Count triple-barrier or meta-label outcomes:

```python
from liq.metrics import summarize_labels

# Labels: 1 = profit, -1 = loss, 0 = neutral/timeout
labels = [1, -1, 0, 1, 1, -1, 0]
counts = summarize_labels(labels)
# {'positive': 3, 'negative': 2, 'neutral': 2}
```

## API Reference

### `summarize_qa(qa_result)`

Convert a QA result object to a flat dictionary.

**Parameters:**
- `qa_result`: A QA result object (dataclass, SimpleNamespace, or dict), or `None`

**Returns:** `dict[str, float | int]` - Flat dictionary with QA metrics

**Raises:** `TypeError` if input is not a supported type

### `summarize_drift(statistics)`

Compute basic statistics for drift signals.

**Parameters:**
- `statistics`: Iterable of numeric drift values (floats or ints)

**Returns:** `dict[str, float]` with keys `'max'` and `'mean'`

**Raises:** `TypeError` if any element is not numeric

### `summarize_labels(labels)`

Count triple-barrier/meta-label outcomes.

**Parameters:**
- `labels`: Iterable of integer labels

**Returns:** `dict[str, int]` with keys `'positive'`, `'negative'`, `'neutral'`

**Raises:** `TypeError` if any element is not an integer

## Type Safety

This package is fully typed and includes a `py.typed` marker for PEP 561 compliance. It exports the `QAResultLike` protocol for type checking QA result objects.

```python
from liq.metrics import QAResultLike

def process_qa(result: QAResultLike) -> None:
    ...
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src tests

# Run type checking
mypy src
```

## License

MIT
