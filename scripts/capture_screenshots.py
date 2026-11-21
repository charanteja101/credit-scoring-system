import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
PORT = 8599
URL = f"http://localhost:{PORT}"

PAGES = [
    ("01_overview", "Executive Overview", []),
    ("02_bi_dashboard", "Portfolio BI Dashboard", [("02b_bi_segments", "Segments"), ("02c_bi_heatmap", "Heatmap")]),
    ("03_model_performance", "Model Performance",
     [("03b_roc_ks", "ROC & KS"), ("03c_lift_gain", "Lift & Gain"), ("03d_calibration", "Calibration")]),
    ("04_scorecard", "Scorecard & Features", [("04b_feature_explorer", "Feature explorer")]),
    ("05_strategy", "Cut-off & Strategy", []),
    ("06_monitoring", "Monitoring & Drift", []),
    ("07_applicant_scoring", "Applicant Scoring", []),
]


def wait_for_server(timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(URL, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Streamlit did not start in time")


def settle(page, ms=4500):
    page.wait_for_timeout(ms)


def shot(page, name):
    path = OUT / f"{name}.png"
    page.screenshot(path=str(path))
    print("saved", path.relative_to(ROOT))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
         "--server.port", str(PORT), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1900})
            page.goto(URL)
            page.wait_for_selector("[data-testid='stSidebar']", timeout=60000)
            settle(page, 6000)
            for name, label, tabs in PAGES:
                try:
                    page.locator("[data-testid='stSidebar'] label", has_text=label).first.click()
                    settle(page)
                    if label == "Applicant Scoring":
                        page.get_by_role("button", name="Calculate score").click()
                        settle(page)
                    shot(page, name)
                    for tab_file, tab_name in tabs:
                        page.get_by_role("tab", name=tab_name).first.click()
                        settle(page, 3000)
                        shot(page, tab_file)
                except Exception as e:
                    print(f"!! skipped {name}: {e}")
            browser.close()
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
