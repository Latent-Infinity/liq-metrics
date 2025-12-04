# liq-metrics
A library within the Latent Infinity Quant (LIQ) ecosystem, `liq-metrics` is used to evaluate strategy performance based on trade history and equity curves from a simulation.

## New utilities

- QA ingestion: `summarize_qa` flattens liq-data QA results for reporting.
- Drift summaries: `summarize_drift` computes basic stats for drift signals from liq-features.
- Label summaries: `summarize_labels` counts triple-barrier/meta-label outcomes.

Usage:
```python
from liq.metrics import summarize_qa, summarize_drift, summarize_labels

qa_summary = summarize_qa(qa_result)
drift_summary = summarize_drift([0.1, 0.2])
label_summary = summarize_labels([1, -1, 0])
```
