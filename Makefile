# Clarity — the commands you will actually type.
#
# A Makefile is the cheapest documentation there is: it is the only kind that stops
# working when it goes out of date. Everything below is what the README describes, and
# `make help` is what a new person runs first.

.DEFAULT_GOAL := help
.PHONY: help setup index ask serve eval test lint demo docker clean

PY := python3
export PYTHONPATH := $(CURDIR)/code

help:  ## show this list
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## create the virtualenv and install everything
	$(PY) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Now: cp .env.example .env and put your key in it."

index:  ## build the Meridian corpus and its vector index
	$(PY) code/meridian/generate.py
	$(PY) code/meridian_index.py --build

ask:  ## ask one question — make ask Q="what was gross margin in 2023 Q1?"
	@test -n "$(Q)" || (echo 'usage: make ask Q="your question"'; exit 2)
	$(PY) code/clarity/v1_0/cli.py ask "$(Q)"

serve:  ## run the API on :8000
	$(PY) code/clarity/v1_0/cli.py serve

eval:  ## run the six-layer suite against the golden set
	$(PY) code/clarity/v1_0/cli.py eval

demo:  ## the eight-minute demonstration
	$(PY) code/clarity/v1_0/cli.py demo

test:  ## the tests — evaluations run only when OPENAI_API_KEY is set
	$(PY) -m pytest tests/ -q

lint:  ## the checks CI runs
	$(PY) -m ruff check code/ tests/
	$(PY) code/_runner.py --check

docker:  ## build and run the whole thing in containers
	docker compose up --build

clean:  ## remove build output and caches
	rm -rf .pytest_cache/
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
