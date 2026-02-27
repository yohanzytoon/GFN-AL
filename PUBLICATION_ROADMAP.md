# Current Milestone Scope


## Included Methods

### 1. Dataset generation

- Sample random terminating states from the official Scrabble environment in
  `alexhernandezgarcia/gflownet`.
- Query the oracle through the local budget-aware proxy.
- Save `dataset.npz` plus the resulting metrics and run summary.

### 2. Supervised baseline

- Use the randomly queried dataset.
- Train a simple MLP regressor to predict scores.
- Report fit quality and search-quality metrics.

### 3. Standard active learning loop

- Start from a small random seed set.
- Fit a Gaussian-process surrogate.
- Sample a random candidate pool.
- Use UCB to pick the next batch of oracle queries.
- Repeat until the budget is exhausted.


## What To Run Right Now

```bash
python experiments/run_dataset.py
python experiments/run_baseline.py dataset.path=outputs/dataset/<run_dir>/dataset.npz
python experiments/run_active.py
```

For fast checks:

```bash
python experiments/run_dataset.py oracle.budget=40 dataset.num_queries=40
python experiments/run_baseline.py dataset.path=outputs/dataset/<run_dir>/dataset.npz baseline.epochs=5
python experiments/run_active.py oracle.budget=40 active.initial_size=10 active.batch_size=5 active.max_rounds=4 active.candidate_pool_size=32 active.surrogate.fit_maxiter=10
```

## Current Deliverable

At this stage, the code should support:
- a defensible methods description
- one baseline result
- one simple active-learning result

That is the correct scope for the current milestone.
