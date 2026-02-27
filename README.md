# GFlowNets + Active Learning for Sample-Efficient Exploration

Production-ready research repository for comparing:
- Random search
- Supervised baseline
- Pure GFlowNet (oracle reward)
- Pure active learning (surrogate + acquisition)
- Hybrid: GFlowNet generation + surrogate filtering + oracle evaluation

Search space: Scrabble-like sequences via the official `gflownet` Scrabble environment.

## Repository Layout

```text
gfn_active_learning_project/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── baseline.yaml
│   ├── active_learning.yaml
│   ├── hybrid.yaml
│   └── experiments/
│       ├── gflownet_scrabble.yaml
│       ├── hybrid_scrabble.yaml
│       ├── comparisons.yaml
│       └── ablations.yaml
├── src/
│   ├── environments/
│   │   └── scrabble_oracle_env.py
│   ├── proxies/
│   │   └── oracle_proxy.py
│   ├── surrogate/
│   │   ├── gp_model.py
│   │   ├── deep_ensemble.py
│   │   └── bnn.py
│   ├── acquisition/
│   │   ├── ucb.py
│   │   ├── ei.py
│   │   └── thompson.py
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_active.py
│   │   └── train_hybrid.py
│   └── utils/
│       ├── metrics.py
│       ├── logging.py
│       └── visualization.py
├── experiments/
│   ├── run_baseline.py
│   ├── run_active.py
│   ├── run_hybrid.py
│   ├── run_comparisons.py
│   ├── run_ablations.py
│   └── export_publication_tables.py
└── tests/
```

## Official GFlowNet Integration

This project extends the official library directly:
- Environment: `gflownet.envs.scrabble.Scrabble` (extended in `ScrabbleOracleEnv`)
- Proxy: `gflownet.proxy.base.Proxy` (extended in `OracleProxy`)
- Agent: official `GFlowNetAgent` from `gflownet.gflownet`
- Hydra workflow: official `train.py`, `eval.py`, `resume.py` are called from `src/training/train_hybrid.py`
- Losses: `trajectorybalance`, `flowmatch`, `detailedbalance`

## Installation

### 1) Python environment

```bash
cd gfn_active_learning_project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 2) Install official GFlowNet repo

This project expects the official clone at `../gflownet` relative to this directory.

```bash
# from gfn_active_learning_project/
pip install -e ../gflownet
```

### 3) Install this project

```bash
pip install -r requirements.txt
pip install -e .
```

Optional convenience commands:

```bash
make compare
make ablate
make tables RUN_DIR=outputs/comparisons/<run_id>
```

## Running Experiments

### Config 1: Supervised baseline

```bash
python experiments/run_baseline.py
```

Override examples:

```bash
python experiments/run_baseline.py seed=1 oracle.budget=500 env.max_length=7
```

### Config 2: Pure GFlowNet (official train/eval workflow)

Pure GFlowNet is run through `experiments/run_comparisons.py` with `method=pure_gflownet` or through direct API usage in `src/training/train_hybrid.py:run_pure_gflownet`.

Minimal command:

```bash
python experiments/run_comparisons.py comparison.methods=[pure_gflownet] comparison.seeds=[0]
```

Direct official training CLI (from `../gflownet`):

```bash
PYTHONPATH=../gfn_active_learning_project/src:$PYTHONPATH \
python train.py env=scrabble_oracle proxy=oracle_proxy gflownet=trajectorybalance loss=trajectorybalance policy=mlp_trajectorybalance
```

### Config 3: Active learning

```bash
python experiments/run_active.py
```

Acquisition overrides:

```bash
python experiments/run_active.py active.acquisition=ei
python experiments/run_active.py active.acquisition=thompson
python experiments/run_active.py active.acquisition=uncertainty
```

Surrogate overrides:

```bash
python experiments/run_active.py active.surrogate_type=gp
python experiments/run_active.py active.surrogate_type=deep_ensemble
python experiments/run_active.py active.surrogate_type=bnn
```

### Config 4: Hybrid (GFlowNet + Active Learning)

```bash
python experiments/run_hybrid.py
```

Key overrides:

```bash
python experiments/run_hybrid.py active.surrogate_type=gp active.acquisition=ucb
python experiments/run_hybrid.py gflownet.objective=flowmatch
python experiments/run_hybrid.py gflownet.objective=detailedbalance
python experiments/run_hybrid.py hybrid.retrain_frequency=1
```

## Resume and Evaluate Official GFlowNet Runs

The training module exposes wrappers for the official scripts:
- `run_gflownet_train(...)` -> `gflownet/train.py`
- `run_gflownet_eval(...)` -> `gflownet/eval.py`
- `run_gflownet_resume(...)` -> `gflownet/resume.py`

Resume a prior run programmatically:

```python
from pathlib import Path
from training.train_hybrid import run_gflownet_resume

run_gflownet_resume(
    repo_root=Path("../gflownet").resolve(),
    project_src=Path("src").resolve(),
    rundir=Path("outputs/hybrid/.../gflownet_runs/round_000").resolve(),
    device="cpu",
    seed=0,
)
```

## Full Comparison + Statistical Tests

Run all methods with multi-seed comparison:

```bash
python experiments/run_comparisons.py
```

Outputs:
- `comparison_results.csv`
- `method_summary.csv` (mean/std/95% CI by method)
- `curve_statistics.csv` (query-wise mean/lower/upper for each method)
- `pairwise_tests.json` (paired t-test + Wilcoxon + Cohen's d)
- `best_vs_queries_ci.png`
- `top10_vs_queries_ci.png`
- `regret_vs_queries_ci.png`

## Reproducing Main Figures

```bash
python experiments/run_comparisons.py comparison.seeds=[0,1,2,3,4]
```

Figures are generated in the Hydra run directory under `outputs/comparisons/...`:
- `best_vs_queries_ci.png`
- `top10_vs_queries_ci.png`
- `regret_vs_queries_ci.png`

## Ablation Sweeps

### Acquisition function

```bash
python experiments/run_comparisons.py comparison.methods=[active_learning,hybrid] base_config.active.acquisition=ucb
python experiments/run_comparisons.py comparison.methods=[active_learning,hybrid] base_config.active.acquisition=ei
python experiments/run_comparisons.py comparison.methods=[active_learning,hybrid] base_config.active.acquisition=thompson
```

### Surrogate type

```bash
python experiments/run_comparisons.py base_config.active.surrogate_type=gp
python experiments/run_comparisons.py base_config.active.surrogate_type=deep_ensemble
python experiments/run_comparisons.py base_config.active.surrogate_type=bnn
```

### GFlowNet temperature

```bash
python experiments/run_comparisons.py base_config.gflownet.temperature_logits=0.7
python experiments/run_comparisons.py base_config.gflownet.temperature_logits=1.0
python experiments/run_comparisons.py base_config.gflownet.temperature_logits=1.5
```

### Batch size K

```bash
python experiments/run_comparisons.py base_config.active.batch_size=8
python experiments/run_comparisons.py base_config.active.batch_size=16
python experiments/run_comparisons.py base_config.active.batch_size=32
```

### Initial dataset size

```bash
python experiments/run_comparisons.py base_config.active.initial_size=32
python experiments/run_comparisons.py base_config.active.initial_size=64
python experiments/run_comparisons.py base_config.active.initial_size=128
```

### Hybrid retrain frequency

```bash
python experiments/run_comparisons.py base_config.hybrid.retrain_frequency=1
python experiments/run_comparisons.py base_config.hybrid.retrain_frequency=2
python experiments/run_comparisons.py base_config.hybrid.retrain_frequency=4
```

## Dedicated Ablation Runner

```bash
python experiments/run_ablations.py
```

Outputs:
- `ablation_results.csv`
- `ablation_summary.csv`
- `ablation_<sweep_name>_best_score.png`

## Export Paper Tables

After a comparisons run, export LaTeX/Markdown tables directly:

```bash
python experiments/export_publication_tables.py --run-dir outputs/comparisons/<run_id>
```

Outputs:
- `tables/table_method_summary.tex`
- `tables/table_method_summary.md`
- `tables/table_pairwise_tests.tex`
- `tables/table_pairwise_tests.md`

## Tests

```bash
pytest -q
```

## Publication Workflow

Use `PUBLICATION_ROADMAP.md` for the end-to-end checklist from experiment execution to paper-ready tables/figures.

## Notes

- Oracle budgets are strictly enforced by `OracleProxy` in oracle mode.
- Hybrid retrains GFlowNet with surrogate rewards while preserving official GFlowNet training code.
- WandB logging is optional and disabled by default.
- Vocabulary validity checks are enabled by default (`oracle.vocabulary_check=true`) for publication-grade validity metrics.
