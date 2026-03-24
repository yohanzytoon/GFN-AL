# GFlowNets + Active Learning for Scrabble Exploration

This repository now covers the code-side scope up to Phase 4 of the project:
- random oracle-labeled dataset generation
- supervised baseline training
- active learning with configurable surrogate and acquisition
- oracle-access GFlowNet training through the upstream `alexhernandezgarcia/gflownet` repo
- hybrid `surrogate -> GFlowNet -> acquisition -> oracle` training
- multi-seed comparison and ablation runners
- result aggregation, confidence-interval curves, and paired statistical tests

## What is in the repo

- `experiments/run_dataset.py`
- `experiments/run_baseline.py`
- `experiments/run_active.py`
- `experiments/run_gflownet.py`
- `experiments/run_hybrid.py`
- `experiments/run_comparisons.py`
- `experiments/run_ablations.py`
- `src/training/dataset.py`
- `src/training/train_baseline.py`
- `src/training/train_active.py`
- `src/training/train_gflownet.py`
- `src/training/train_hybrid.py`
- `src/environments/scrabble_oracle_env.py`
- `src/proxies/oracle_proxy.py`
- `src/proxies/surrogate_proxy.py`
- `src/surrogate/gp_model.py`
- `src/surrogate/ensemble_model.py`
- `src/acquisition/ucb.py`
- `src/acquisition/ei.py`
- `src/acquisition/thompson.py`
- `src/acquisition/uncertainty.py`
- `src/utils/results.py`
- `src/utils/visualization.py`

## Required External Dependency

The Scrabble environment and the upstream GFlowNet training stack come from:

- `https://github.com/alexhernandezgarcia/gflownet`

This project expects that repository as a sibling checkout at `../gflownet`.

```bash
cd /Users/youhannazytoon/gflownet
git -C gflownet remote add upstream https://github.com/alexhernandezgarcia/gflownet.git
git -C gflownet remote -v
```

## Setup

Use Python `3.11` or `3.12`.

```bash
cd /Users/youhannazytoon/GFN-AL
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -e . -e ../gflownet
```

## What To Run

Dataset generation:

```bash
.venv/bin/python experiments/run_dataset.py
```

Baseline:

```bash
.venv/bin/python experiments/run_baseline.py dataset.path=outputs/dataset/<run_dir>/dataset.npz
```

Active learning:

```bash
.venv/bin/python experiments/run_active.py
```

Oracle GFlowNet:

```bash
.venv/bin/python experiments/run_gflownet.py
```

Hybrid:

```bash
.venv/bin/python experiments/run_hybrid.py
```

Comparisons:

```bash
.venv/bin/python experiments/run_comparisons.py
```

Ablations:

```bash
.venv/bin/python experiments/run_ablations.py
```

Short smoke runs:

```bash
.venv/bin/python experiments/run_dataset.py oracle.budget=40 dataset.num_queries=40
.venv/bin/python experiments/run_baseline.py dataset.path=outputs/dataset/<run_dir>/dataset.npz baseline.epochs=5
.venv/bin/python experiments/run_active.py oracle.budget=40 active.initial_size=10 active.batch_size=5 active.max_rounds=4 active.candidate_pool_size=32 active.surrogate.type=ensemble active.surrogate.epochs=3
.venv/bin/python experiments/run_gflownet.py oracle.budget=40 oracle.vocabulary_check=false gflownet.n_train_steps=10 gflownet.batch_size_forward=4
.venv/bin/python experiments/run_hybrid.py oracle.budget=40 oracle.vocabulary_check=false hybrid.initial_size=8 hybrid.batch_size=4 hybrid.max_rounds=2 hybrid.gflownet.n_train_steps=5
```

`run_baseline.py` still expects a saved `.npz` dataset produced by `run_dataset.py`.

## Validation Status

- Unit tests: `24 passed`
- Real smoke runs completed for:
  - `experiments/run_gflownet.py`
  - `experiments/run_hybrid.py`
