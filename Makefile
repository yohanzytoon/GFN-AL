PYTHON ?= python3

.PHONY: install test dataset baseline active

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

dataset:
	$(PYTHON) experiments/run_dataset.py

baseline:
	$(PYTHON) experiments/run_baseline.py

active:
	$(PYTHON) experiments/run_active.py
