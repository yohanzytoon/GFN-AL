PYTHON ?= .venv/bin/python

.PHONY: install test active gflownet hybrid

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -e ../gflownet

test:
	$(PYTHON) -m pytest -q

active:
	$(PYTHON) experiments/run_active.py

gflownet:
	$(PYTHON) experiments/run_gflownet.py

hybrid:
	$(PYTHON) experiments/run_hybrid.py
