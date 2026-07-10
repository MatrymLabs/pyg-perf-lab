.PHONY: env fix lint typecheck test check bench-cpu clean

# --- Environment: dev tools only (no torch -- that is the heavy `bench` extra) ---
env:
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -e ".[dev]"
	@echo "✓ dev env ready. For benchmarks: install torch for your platform, then pip install -e '.[bench]'"

fix:
	ruff format .
	ruff check . --fix

# --- Gates (run without torch; the bench extra is optional) ---
lint:
	ruff format --check .
	ruff check .

typecheck:
	mypy src tests

test:
	pytest -q

check: lint typecheck test

# --- Run the CPU benchmark for real (needs the bench extra installed) ---
bench-cpu:
	python scripts/bench_loader.py --device cpu --num-nodes 5000 --steps 15

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache *.egg-info src/*.egg-info trace_*.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
