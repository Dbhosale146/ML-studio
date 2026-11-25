import { expect, test, type Page } from '@playwright/test'
import { fileURLToPath } from 'node:url'
import { readFileSync } from 'node:fs'

const CHURN = fileURLToPath(new URL('../../../../data/samples/customer_churn.xlsx', import.meta.url))

/** Answer every question the agent asks with its default/first option until the run finishes. */
async function finishRun(page: Page) {
  const answered = new Set<string>()
  const done = page.getByRole('button', { name: 'Test this model' })
  const deadline = Date.now() + 5 * 60_000
  while (Date.now() < deadline) {
    if (await done.isVisible()) return
    const ask = page.getByTestId('ask')
    if (await ask.isVisible()) {
      const id = (await ask.getAttribute('data-ask-id')) ?? ''
      if (!answered.has(id)) {
        answered.add(id)
        // the sheet picker marks the default; other asks put the recommended answer first
        const kind = await ask.getAttribute('data-ask-kind')
        const target = kind === 'sheet' ? ask.getByRole('button').filter({ hasText: 'Largest' }) : ask.getByRole('button').first()
        await target.click()
      }
    }
    await expect(page.getByRole('alert')).toHaveCount(0)
    await page.waitForTimeout(500)
  }
  throw new Error('agent run did not finish in time')
}

test('upload churn → agent builds a model → single prediction → batch download', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(String(e)))

  await page.goto('/agent')
  await page.getByLabel('Your goal').fill('Predict which customers will churn')
  await page.locator('input[type=file]').first().setInputFiles(CHURN)
  await expect(page.getByText(/Using\s+customer_churn\.xlsx/)).toBeVisible()
  await page.getByRole('button', { name: 'Start' }).click()
  await expect(page).toHaveURL(/\/agent\/ses_/)

  await finishRun(page)
  // the critic caught the leaky column and the run published a model
  await expect(page.getByText('cancellation_reason').first()).toBeVisible()
  await page.getByRole('button', { name: 'Test this model' }).click()
  await expect(page).toHaveURL(/\/models\/.+\?tab=test/)
  await expect(page.getByText('Ready').first()).toBeVisible()

  // single prediction with drivers
  await page.getByRole('button', { name: 'Random training row' }).click()
  await page.getByRole('button', { name: 'Predict', exact: true }).click()
  await expect(page.getByText('Top drivers for this prediction')).toBeVisible()

  // batch scoring and download
  await page.getByRole('tab', { name: /Batch/ }).click()
  await page.locator('input[type=file]').setInputFiles(CHURN)
  const link = page.getByRole('link', { name: 'Download .xlsx' })
  await expect(link).toBeVisible({ timeout: 60_000 })
  const [download] = await Promise.all([page.waitForEvent('download'), link.click()])
  expect(download.suggestedFilename()).toBe('customer_churn_predictions.xlsx')
  const bytes = readFileSync(await download.path())
  expect(bytes.subarray(0, 2).toString()).toBe('PK') // an .xlsx (zip) file
  expect(bytes.length).toBeGreaterThan(10_000)
  await expect(page.getByText(/rows scored/)).toBeVisible()

  expect(errors).toEqual([])
})
