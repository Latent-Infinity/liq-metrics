# liq-metrics

Performance metrics and evaluation utilities for the LIQ Stack ecosystem.

## Installation

```bash
uv pip install liq-metrics
```

For development:
```bash
uv pip install liq-metrics[dev]
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

## Research metrics panel

Every research run reports one shared panel written to `metrics_panel.csv`
(long format `field,value`; every field is emitted even when unavailable):

```python
from liq.metrics import InferenceInputs, compute_metrics_panel, write_metrics_panel_csv

panel = compute_metrics_panel(
    trade_returns_net=net_trades,
    trade_returns_gross=gross_trades,
    daily_returns=daily,
    benchmark_daily_returns=spy_daily,          # alpha/beta attribution
    inference=InferenceInputs(deflated_sharpe=dsr, pbo=pbo),  # from liq-validation
    cost_stress={"spy_qqq_stress_3x_v1": stressed_net},
)
write_metrics_panel_csv(panel, run_dir / "metrics_panel.csv")
```

Inference statistics (bootstrap CI, clustered t-stats, DSR, PBO/null
percentile) are computed with `liq.validation.stats` and passed in — this
package stays dependency-free.

To satisfy paired raw/residual reporting without mixing evidence between the
two panels, residualize in the research layer and supply both aligned outcome
sets explicitly:

```python
from liq.metrics import compute_raw_and_residualized_panels

panels = compute_raw_and_residualized_panels(
    trade_returns_net=raw_net,
    trade_returns_gross=raw_gross,
    daily_returns=raw_daily,
    residualized_trade_returns_net=residual_net,
    residualized_trade_returns_gross=residual_gross,
    residualized_daily_returns=residual_daily,
    benchmark_daily_returns=spy_daily,
    raw_inference=raw_inference,
    residualized_inference=residual_inference,
)
```

Raw and residualized inference, cost-stress, and portfolio-attribution inputs
are separate by design; the helper never copies raw evidence into the
residualized panel.

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

## Dividend receivables

`liq.metrics.entitlements` provides immutable `CashRates` and `Entitlement` facts
and a composable `EntitlementBook`. Upstream callers supply verified eligibility,
held-unit basis, legal-units conversion, explicit gross/net rates, UTC clocks and
provenance. Missing rates remain unavailable; no tax, fee or FX value is inferred.

Accrual must occur at the effective entitlement timestamp and returns zero cash
movement. Late corrections are unsupported. Known receivables contribute to `value(at=...)`
until explicit settlement, including after a sale or at study end. Settlement returns
the amount the cash ledger must credit exactly once. Duplicate accrual/settlement is
rejected, and timestamped history preserves past valuations. Currency/basis mismatch
or unavailable required facts raise `IncompleteEntitlements` with action identities.
The component owns no cash or positions and does not authorize source admission.

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src tests

# Run type checking
mypy src
```

## License

MIT
