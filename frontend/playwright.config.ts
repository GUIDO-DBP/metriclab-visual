import { defineConfig } from '@playwright/test'
export default defineConfig({
  testDir: './e2e',
  use: { baseURL: process.env.PLAYWRIGHT_BASE_URL || 'http://127.0.0.1:8000', channel: 'msedge', headless: true },
  workers: 1,
  timeout: 60000,
  reporter: 'list',
  outputDir: '../runtime/browser-tests'
})
