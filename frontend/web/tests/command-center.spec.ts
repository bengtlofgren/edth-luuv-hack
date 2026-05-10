import { expect, test } from "@playwright/test";

test("command center renders core operator surfaces", async ({ page }) => {
  await page.goto("/");

  await expect(page.locator("#map")).toBeVisible();
  await expect(page.locator("#sidebar")).toBeVisible();
  await expect(page.locator("#gpsdeny-btn")).toBeVisible();
  await expect(page.locator("#mode-select")).toBeVisible();
  await expect(page.locator("#event-log")).toBeVisible();

  const health = await page.evaluate(async () => {
    const response = await fetch("/api/health");
    return response.json();
  });
  expect(["ok", "degraded", "critical"]).toContain(health.status);

  await page.screenshot({ path: "test-results/command-center.png", fullPage: true });
});

test("embedded planner route is served by command center", async ({ page }) => {
  await page.goto("/planner/");
  await expect(page.locator("body")).toBeVisible();
});
