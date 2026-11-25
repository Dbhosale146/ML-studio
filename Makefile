# Datalyon AI Data Scientist: developer commands.
# On Windows without GNU make, use the equivalent PowerShell script: ./make.ps1 <target>
API := services/api
WEB := apps/web

.PHONY: setup dev dev-api dev-web test test-api test-web lint lint-api lint-web format typecheck gen-types samples e2e clean

setup:
	cd $(API) && uv sync
	cd $(WEB) && npm ci
	[ -f .env ] || cp .env.example .env

dev:
	$(MAKE) -j2 dev-api dev-web

dev-api:
	cd $(API) && uv run uvicorn datalyon.main:app --reload --reload-dir src --port 8000

dev-web:
	cd $(WEB) && npm run dev

test: test-api test-web

test-api:
	cd $(API) && uv run pytest --cov-fail-under=80

test-web:
	cd $(WEB) && npm test

lint: lint-api lint-web

lint-api:
	cd $(API) && uv run ruff check src tests && uv run ruff format --check src tests

lint-web:
	cd $(WEB) && npm run lint

format:
	cd $(API) && uv run ruff format src tests && uv run ruff check --fix src tests
	cd $(WEB) && npm run format

typecheck:
	cd $(API) && uv run mypy
	cd $(WEB) && npm run typecheck

gen-types:
	cd $(API) && uv run python -m datalyon.export_openapi openapi.json
	cd $(WEB) && npm run gen-types

samples:
	cd $(API) && uv run python ../../scripts/generate_samples.py

e2e:
	cd $(WEB) && npm run e2e

clean:
	rm -rf storage/uploads storage/artifacts storage/outputs storage/mlruns storage/datalyon.db
