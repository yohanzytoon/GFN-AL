PYTHON ?= python3

.PHONY: install test compare ablate tables

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

compare:
	$(PYTHON) experiments/run_comparisons.py

ablate:
	$(PYTHON) experiments/run_ablations.py

# Usage: make tables RUN_DIR=outputs/comparisons/<run_id>
tables:
	$(PYTHON) experiments/export_publication_tables.py --run-dir $(RUN_DIR)
