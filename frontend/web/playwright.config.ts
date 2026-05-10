import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  use: {
    baseURL: "http://127.0.0.1:8010",
    trace: "on-first-retry",
  },
  webServer: {
    command:
      "cd ../.. && PY=python && [ -x .venv/bin/python ] && PY=.venv/bin/python; DISABLE_SIMULATOR=1 $PY -m uvicorn src.server:app --host 127.0.0.1 --port 8010",
    url: "http://127.0.0.1:8010/api/state",
    reuseExistingServer: false,
    timeout: 20_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
