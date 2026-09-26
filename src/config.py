from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "data" / "results"

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
           "TSLA", "ORCL", "ADBE", "CRM", "INTC", "AMD"]
BENCHMARK = "QQQ"

START = "2022-11-01"
END = "2026-09-26"
