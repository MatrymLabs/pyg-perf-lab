.PHONY: env fix lint typecheck test check security secrets sbom patch bench-cpu clean

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
	pytest -q --cov=pyg_perf --cov-report=term-missing

check: lint typecheck test

# --- Security gate (matches the fleet control panel): SAST + deps + secrets ---
# Report-only; never mutates. Run in CI so vuln status is verified, not assumed.
security:
	bandit -q -r src scripts
	pip-audit --skip-editable
	@git ls-files | detect-secrets-hook --baseline .secrets.baseline

secrets:
	@git ls-files | detect-secrets-hook --baseline .secrets.baseline

# --- Software bill of materials (CycloneDX) from the installed environment ---
# Supply-chain evidence (NIST SSDF PS.3): git-ignored (reproducible from the recorded
# commit), produced + uploaded as an artifact in CI. Advertised, not committed.
sbom:
	@mkdir -p security-evidence
	cyclonedx-py environment -o security-evidence/sbom.cdx.json
	@echo "✓ SBOM -> security-evidence/sbom.cdx.json"

# --- Security patch cycle: scan deps -> file dated evidence -> apply fixes -> re-verify ---
# Auto-applies to the venv only; never commits or pushes a bump.
patch:
	@mkdir -p security-evidence
	@echo "→ scanning dependencies for known CVEs..."
	-pip-audit --skip-editable -f json -o "security-evidence/$$(date -u +%Y-%m-%d)-pip-audit.json"
	@echo "→ applying available security fixes (pip-audit --fix)..."
	-pip-audit --fix --skip-editable
	@echo "→ re-verifying the patched environment..."
	$(MAKE) check
	@echo "✓ security patch cycle complete (evidence: security-evidence/)"

# --- Run the CPU benchmark for real (needs the bench extra installed) ---
bench-cpu:
	python scripts/bench_loader.py --device cpu --num-nodes 5000 --steps 15

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache *.egg-info src/*.egg-info trace_*.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
