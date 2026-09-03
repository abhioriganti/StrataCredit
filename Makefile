# StrataCredit — Makefile
# All commands should be run from the repository root.

.PHONY: help install install-dev lint format typecheck \
 fixture test test-unit test-data test-regression test-integration \
 ingest validate transform analytics train xgb score severity survival-credit survival-prepay mlflow leakage-audit backtest pool app \
 pipeline clean clean-bronze clean-silver clean-gold clean-models \
 docs

PYTHON ?= python
PIP ?= pip
SRC = src/stratacredit
TESTS = tests

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Environment ──────────────────────────────────────────────────────────────

install: ## Install core dependencies
	$(PIP) install -e .

install-dev: ## Install all dependencies including dev extras
	$(PIP) install -e ".[dev,optimization]"

# ─── Code Quality ─────────────────────────────────────────────────────────────

lint: ## Run ruff linter
	ruff check $(SRC) $(TESTS)

format: ## Run ruff formatter
	ruff format $(SRC) $(TESTS)

format-check: ## Check formatting without modifying
	ruff format --check $(SRC) $(TESTS)

typecheck: ## Run mypy type checker
	mypy $(SRC)

# ─── Testing ──────────────────────────────────────────────────────────────────

fixture: ## Generate synthetic test fixtures
	$(PYTHON) -m stratacredit.fixtures.generate

test: ## Run all tests
	pytest $(TESTS) -m "not integration"

test-unit: ## Run unit tests only
	pytest $(TESTS)/unit -m unit

test-data: ## Run data validation tests
	pytest $(TESTS)/data -m data

test-regression: ## Run golden fixture regression tests
	pytest $(TESTS)/regression -m regression

test-integration: ## Run full integration tests (slow)
	pytest $(TESTS)/integration -m integration

test-smoke: ## Run CI smoke tests
	pytest $(TESTS) -m smoke --tb=short

# ─── Pipeline Steps ───────────────────────────────────────────────────────────

ingest: ## Ingest raw files → Bronze Parquet
	$(PYTHON) -m stratacredit.ingest.runner

validate: ## Run data quality validation
	$(PYTHON) -m stratacredit.quality.runner

transform: ## Build Silver tables (origination, performance, panel)
	$(PYTHON) -m stratacredit.transform.runner

analytics: ## Build Gold analytical tables
	$(PYTHON) -m stratacredit.analytics.runner

train: ## Train full-universe logistic baseline models
	$(PYTHON) -m stratacredit.models.full_trainer

xgb: ## Train temporally validated XGBoost challenger models
	$(PYTHON) -m stratacredit.models.xgb_trainer

score: ## Persist OOT loan-level model scores and portfolio backtests
	$(PYTHON) -m stratacredit.models.scoring

severity: ## Train conditional loss-severity model
	$(PYTHON) -m stratacredit.models.severity_trainer

survival-credit: ## Train right-censored credit-event AFT model
	$(PYTHON) -m stratacredit.models.survival_trainer

survival-prepay: ## Train right-censored prepayment AFT model
	$(PYTHON) -m stratacredit.models.survival_prepay_trainer

leakage-audit: ## Run model feature and temporal leakage diagnostics
	$(PYTHON) -m stratacredit.models.leakage_audit

mlflow: ## Register persisted models and diagnostics in local MLflow
	$(PYTHON) -m stratacredit.models.mlflow_tracking

backtest: ## Run out-of-time backtests
	$(PYTHON) -m stratacredit.models.backtest

pool: ## Run pool selection
	$(PYTHON) -m stratacredit.optimization.runner

scenarios: ## Run stress scenario analysis
	$(PYTHON) -m stratacredit.scenarios.runner

report: ## Generate analytical memo and exports
	$(PYTHON) -m stratacredit.reporting.runner

app: ## Launch Streamlit analyst workbench
	streamlit run app/streamlit_app.py

# ─── End-to-end Pipeline ──────────────────────────────────────────────────────

pipeline: ingest validate transform analytics ## Full pipeline: Bronze → Silver → Gold
	@echo "Pipeline complete. Run 'make train' for modeling."

all: pipeline train xgb score backtest pool scenarios report ## Complete analytical workflow

# ─── Cleanup ──────────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage

clean-bronze: ## Remove Bronze Parquet files
	rm -rf data/bronze/

clean-silver: ## Remove Silver DuckDB/Parquet files
	rm -rf data/silver/

clean-gold: ## Remove Gold analytical tables
	rm -rf data/gold/

clean-models: ## Remove trained model artifacts
	rm -rf mlruns/ models/

clean-all: clean clean-bronze clean-silver clean-gold clean-models ## Remove all generated data

# ─── Documentation ────────────────────────────────────────────────────────────

docs: ## Open documentation index
	@echo "Documentation available in docs/"
	@ls docs/
