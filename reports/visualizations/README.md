# Report Visualizations

This folder contains report-ready figures generated from the latest comparison run.

Current main run:
- `outputs/comparisons/2026-03-29_21-12-32`

Reference baseline run:
- `outputs/comparisons/2026-03-29_14-51-09`

Generate / refresh the figures with:

```bash
.venv/bin/python reports/visualizations/generate_report_figures.py
```

Generated assets:
- `assets/quality_overview.png`
- `assets/cost_tradeoffs.png`
- `assets/query_dynamics.png`
- `assets/report_metrics_summary.csv`
