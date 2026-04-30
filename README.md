# GFN Active Learning

Scrabble search experiments with three tracked entry points:

- `experiments/run_active.py`: GP surrogate active learning with UCB.
- `experiments/run_gflownet.py`: direct oracle-trained GFlowNet baseline.
- `experiments/run_hybrid.py`: hybrid GP + GFlowNet active learning.

The Scrabble environment and scorer come from the upstream GFlowNet repository.
This repo expects that checkout at `../gflownet`, or an explicit
`gflownet_root` override in the Hydra config.

## Upstream GFlowNet dependency

This project depends on [alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet) for the Scrabble environment and scorer.
Clone it **as a sibling directory** so it lives at `../gflownet` relative to this repo:

```bash
git clone https://github.com/alexhernandezgarcia/gflownet.git ../gflownet
```

If you prefer a different location, set `gflownet_root` in your Hydra config to point at it.

The gflownet repo requires **Python 3.10** and PyTorch 2.5.1.



## Setup

Use Python 3.11 or 3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e ../gflownet
```

## Run

```bash
python experiments/run_active.py
python experiments/run_gflownet.py
python experiments/run_hybrid.py
```

Short smoke examples:

```bash
python experiments/run_active.py oracle.budget=40 active.initial_size=10 active.batch_size=5 active.max_rounds=4 active.candidate_pool_size=32 active.surrogate.fit_maxiter=10 active.surrogate.prefer_botorch=false
python experiments/run_hybrid.py oracle.budget=40 hybrid.initial_size=8 hybrid.batch_size=4 hybrid.max_rounds=2 hybrid.gflownet.n_train_steps=10 hybrid.gflownet.sample_size=8 hybrid.surrogate.fit_maxiter=10 hybrid.surrogate.prefer_botorch=false
```

## Test

```bash
python -m pytest -q
```

## Makefile

All commands use `.venv/bin/python` by default. Override with `PYTHON=<path> make <target>`.

| Target | Command |
|---|---|
| `make install` | Install pip deps, this package, and `../gflownet` in editable mode |
| `make test` | Run the test suite (`pytest -q`) |
| `make active` | Run the active learning loop via `run_active.py` |
| `make gflownet` | Run the GFlowNet experiment via `run_gflownet.py` |
| `make hybrid` | Run the hybrid experiment via `run_hybrid.py` |


