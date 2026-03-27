# Preliminary Milestone Repo

This branch is now restricted to the preliminary scope only:
- generate a dataset with the official Scrabble environment
- train a supervised baseline on a saved random oracle-labeled dataset
- run one standard active-learning loop (GP surrogate + UCB)


## What do we have

- `experiments/run_baseline.py`
- `experiments/run_dataset.py`
- `experiments/run_active.py`
- `src/training/dataset.py`
- `src/training/train_baseline.py`
- `src/training/train_active.py`
- `src/environments/scrabble_oracle_env.py`
- `src/proxies/oracle_proxy.py`
- `src/surrogate/gp_model.py`
- `src/acquisition/ucb.py`

## Required External Dependency

The Scrabble environment and Scrabble scorer still come from the official repo:

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
cd /Users/youhannazytoon/gflownet/GFN-AL
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## What To Run

Dataset generation:

```bash
python experiments/run_dataset.py
```

Baseline:

```bash
python experiments/run_baseline.py dataset.path=outputs/dataset/<run_dir>/dataset.npz
```

Active learning:

```bash
python experiments/run_active.py
```

Short smoke runs:

```bash
python experiments/run_dataset.py oracle.budget=40 dataset.num_queries=40
python experiments/run_baseline.py dataset.path=outputs/dataset/<run_dir>/dataset.npz baseline.epochs=5
python experiments/run_active.py oracle.budget=40 active.initial_size=10 active.batch_size=5 active.max_rounds=4 active.candidate_pool_size=32 active.surrogate.fit_maxiter=10
```

`run_baseline.py` no longer samples data on its own. It expects a saved `.npz`
dataset produced by `run_dataset.py`.

## Current Preliminary Evidence

One baseline run has already succeeded locally and produced:
- best score: `11.0`
- top-10 average score: `8.0`
- valid word ratio: `0.041`

This is enough for the current methods/preliminary-results milestone.
