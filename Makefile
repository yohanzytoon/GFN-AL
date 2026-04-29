PYTHON ?= .venv/bin/python

.PHONY: install test dataset baseline active gflownet hybrid comparisons ablations quick-compare full-compare

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -e ../gflownet

test:
	$(PYTHON) -m pytest -q

dataset:
	$(PYTHON) experiments/run_dataset.py

baseline:
	$(PYTHON) experiments/run_baseline.py

active:
	$(PYTHON) experiments/run_active.py

gflownet:
	$(PYTHON) experiments/run_gflownet.py

hybrid:
	$(PYTHON) experiments/run_hybrid.py

comparisons:
	$(PYTHON) experiments/run_comparisons.py

ablations:
	$(PYTHON) extras/run_ablations.py

quick-compare:
	./scripts/run_compare.sh quick

full-compare:
	./scripts/run_compare.sh full
