import { defineConfig, devices } from '@playwright/test'

// Starts the API (offline MockLLM, isolated storage) and the Vite dev server, or reuses ones already running.
const apiEnv = {
  LLM_PROVIDER: 'mock',
  MLFLOW_ENABLED: 'false',
  MLFLOW_DISABLE_AGENT_HINT: '1',
  TRAIN_TIME_BUDGET_S: '60',
  STORAGE_DIR: '../../storage/e2e',
}

export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 6 * 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    acceptDownloads: true,
    ...(process.env.PW_CHANNEL ? { channel: process.env.PW_CHANNEL } : {}),
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'], viewport: { width: 1440, height: 950 } } }],
  webServer: [
    {
      command: 'uv run uvicorn datalyon.main:app --port 8000',
      cwd: '../../services/api',
      url: 'http://localhost:8000/api/v1/health',
      env: apiEnv,
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: 'npm run dev -- --port 5173 --strictPort',
      url: 'http://localhost:5173',
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
})
