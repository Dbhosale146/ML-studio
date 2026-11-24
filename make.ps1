# Windows twin of the Makefile: ./make.ps1 setup | dev | test | lint | format | typecheck | gen-types | samples | e2e | clean
param([Parameter(Position = 0)][string]$Target = 'help')
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Api = Join-Path $Root 'services/api'
$Web = Join-Path $Root 'apps/web'

function In($dir, [scriptblock]$block) { Push-Location $dir; try { & $block; if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" } } finally { Pop-Location } }

switch ($Target) {
  'setup' {
    In $Api { uv sync }
    In $Web { npm ci }
    if (-not (Test-Path (Join-Path $Root '.env'))) { Copy-Item (Join-Path $Root '.env.example') (Join-Path $Root '.env') }
  }
  'dev' {
    $api = Start-Process -PassThru -NoNewWindow -WorkingDirectory $Api uv -ArgumentList 'run', 'uvicorn', 'datalyon.main:app', '--reload', '--reload-dir', 'src', '--port', '8000'
    try { In $Web { npm run dev } } finally { Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue }
  }
  'dev-api' { In $Api { uv run uvicorn datalyon.main:app --reload --reload-dir src --port 8000 } }
  'dev-web' { In $Web { npm run dev } }
  'test' { In $Api { uv run pytest }; In $Web { npm test } }
  'test-api' { In $Api { uv run pytest } }
  'test-web' { In $Web { npm test } }
  'lint' { In $Api { uv run ruff check src tests; uv run ruff format --check src tests }; In $Web { npm run lint } }
  'format' { In $Api { uv run ruff format src tests; uv run ruff check --fix src tests }; In $Web { npm run format } }
  'typecheck' { In $Api { uv run mypy }; In $Web { npm run typecheck } }
  'gen-types' { In $Api { uv run python -m datalyon.export_openapi openapi.json }; In $Web { npm run gen-types } }
  'samples' { In $Api { uv run python ../../scripts/generate_samples.py } }
  'e2e' { In $Web { npm run e2e } }
  'clean' { foreach ($p in 'uploads', 'artifacts', 'outputs', 'mlruns', 'datalyon.db') { Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $Root "storage/$p") } }
  default { Write-Host 'Targets: setup, dev, dev-api, dev-web, test, test-api, test-web, lint, format, typecheck, gen-types, samples, e2e, clean' }
}
