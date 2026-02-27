# Publication Roadmap

## 1) Reproducible Experiment Matrix

Run the full comparison matrix (5 methods x >=5 seeds):

```bash
python experiments/run_comparisons.py
```

Artifacts produced:
- `comparison_results.csv`
- `method_summary.csv`
- `curve_statistics.csv`
- `pairwise_tests.json`
- `best_vs_queries_ci.png`
- `top10_vs_queries_ci.png`
- `regret_vs_queries_ci.png`

## 2) Required Ablations

Run all configured ablations:

```bash
python experiments/run_ablations.py
```

Artifacts produced:
- `ablation_results.csv`
- `ablation_summary.csv`
- `ablation_<sweep_name>_best_score.png`

Sweeps include:
- acquisition function
- surrogate type
- GFlowNet temperature
- batch size
- initial dataset size
- hybrid retrain frequency

## 3) Metrics Available for Paper Tables

Per-run metrics:
- `best_score`
- `top10_score`
- `mean_score`
- `valid_word_ratio`
- `diversity_entropy`
- `unique_fraction`
- `mode_coverage`
- `topk_edit_distance`
- `queries_to_90pct_best`
- `oracle_queries`
- `simple_regret` (if `metrics.optimum_score` is set)

Statistical tests:
- paired t-test
- Wilcoxon signed-rank
- Cohen's d (paired)

## 4) Paper Figure Checklist

Minimum figure set:
- Best score vs oracle queries (95% CI)
- Top-10 score vs oracle queries (95% CI)
- Regret vs oracle queries (95% CI)
- Ablation figure per sweep
- Surrogate fit quality trends (`surrogate_rmse` across rounds)

## 4.1) Table Export

Generate LaTeX tables for report insertion:

```bash
python experiments/export_publication_tables.py --run-dir outputs/comparisons/<run_id>
```

## 5) Configuration Hygiene

Default configs are in:
- `configs/baseline.yaml`
- `configs/active_learning.yaml`
- `configs/hybrid.yaml`

Use `oracle.vocabulary_check=true` for validity-aware experiments.

Optional true-regret setup:
- set `metrics.optimum_score=<known optimum>` in config.

## 6) Final Pre-Submission QA

Before final writeup:
1. Re-run all final experiments from clean output directories.
2. Verify all methods have the same oracle budget and seed list.
3. Verify all plots/tables are generated from saved CSV artifacts.
4. Freeze config overrides used for the final paper in an appendix table.
5. Run tests:

```bash
python -m pytest -q
```

## 7) Stretch Extensions (if time remains)

- Add second environment (e.g., Tetris or bitstring)
- Add uncertainty calibration metrics (ECE/NLL) for surrogates
- Add novelty-diversity aware acquisition function
- Add compute-cost breakdown (wall-time per method)
