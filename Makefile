# LLM Gateway — build / run / test entrypoints.
# `make` runs in WSL and in CI. On Windows use `.\dev.ps1` (same targets).

.DEFAULT_GOAL := help
PY := .venv/Scripts/python.exe            # Windows venv (override: make PY=python)
PIP := $(PY) -m pip

.PHONY: help infra dev test lint fmt bench train seed clean

help:                                   ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

infra:                                  ## start Redis + Postgres inside WSL
	wsl -d Ubuntu -- bash -lc "sudo service redis-server start; redis-cli ping"

venv:                                   ## create the virtualenv
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PIP) install -r requirements-dev.txt

dev:                                    ## run the API + dashboard (two terminals)
	$(PY) -m uvicorn gateway.main:app --reload --port 8000
	@echo "dashboard: cd dashboard && npm run dev"

test:                                   ## run the test suite with coverage
	$(PY) -m pytest -q --cov=gateway --cov-report=term-missing

lint:                                   ## ruff check
	$(PY) -m ruff check gateway tests

fmt:                                    ## ruff format
	$(PY) -m ruff format gateway tests

bench:                                  ## Locust load test (Day 2)
	$(PY) -m locust -f loadtest/locustfile.py --headless -u 100 -t 60s --csv loadtest/results/run

train:                                  ## train the router (Day 2)
	$(PY) ml/train_router.py

seed:                                   ## create an admin user + first API key
	$(PY) -m gateway.cli seed

clean:                                  ## remove runtime artifacts
	rm -rf .pytest_cache .ruff_cache htmlcov gateway.db gateway.db-wal gateway.db-shm
