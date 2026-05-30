import os
import testmu
from testmu import expect, var, set_var
from playwright.async_api import Page

testmu.configure(
    build="2b39f209-fa72-42ee-84bf-57cbb58f8f73",
    name="Web || sahilbasera || TC-4",
    tc_id="TC-4",
    network=True,
    default_action_timeout_ms=10000,
    default_navigation_timeout_ms=30000,
)

@testmu.test
async def test(page: Page):
    async with testmu.step('Navigate to http://localhost:8501'):
        await page.goto("http://localhost:8501")
    
    async with testmu.step('Waiting for the app UI to load'):
        await page.wait_for_timeout(2000)
    
    async with testmu.step('Clicking Browse files to upload the required image'):
        _loc_1 = page.locator("internal:testid=[data-testid=\"stBaseButton-secondary\"s]")
        
        await _loc_1.click()


if __name__ == "__main__":
    testmu.run(test)