"""Find Motley Fool earnings-call transcript URLs for our tickers via the monthly sitemaps.

Sitemap pages are cached in data/raw/sitemaps/ so each month is fetched once.
Writes data/raw/transcript_urls.csv and prints the events found per ticker.
"""
import re
import time

import pandas as pd
import requests

from config import RAW, TICKERS

HEADERS = {"User-Agent": "Mozilla/5.0 (student research project; hughconboy@gmail.com)"}
SITEMAP_DIR = RAW / "sitemaps"
FIRST_MONTH = "2022-12"
LAST_MONTH = "2026-09"
DELAY_SECONDS = 3

# Fool tags Alphabet as either GOOG or GOOGL
TICKER_ALIASES = {"GOOGL": ["googl", "goog"]}
LINK_RE = re.compile(r"/earnings/call-transcripts/(\d{4})/(\d{2})/(\d{2})/([^\"<\s/]+)/?")


def month_range():
    return [p.strftime("%Y/%m") for p in pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M")]


def fetch_month(ym):
    cache = SITEMAP_DIR / (ym.replace("/", "-") + ".html")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    resp = requests.get(f"https://www.fool.com/sitemap/{ym}", headers=HEADERS, timeout=60)
    resp.raise_for_status()
    cache.write_text(resp.text, encoding="utf-8")
    time.sleep(DELAY_SECONDS)
    return resp.text


def match_ticker(slug):
    for ticker in TICKERS:
        for alias in TICKER_ALIASES.get(ticker, [ticker.lower()]):
            if f"-{alias}-q" in slug:
                return ticker
    return None


def main():
    SITEMAP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ym in month_range():
        html = fetch_month(ym)
        for y, m, d, slug in set(LINK_RE.findall(html)):
            ticker = match_ticker(slug)
            if ticker:
                rows.append({"ticker": ticker, "url_date": f"{y}-{m}-{d}", "slug": slug,
                             "url": f"https://www.fool.com/earnings/call-transcripts/{y}/{m}/{d}/{slug}/"})
    df = pd.DataFrame(rows).drop_duplicates("url").sort_values(["ticker", "url_date"])
    df.to_csv(RAW / "transcript_urls.csv", index=False)
    print(f"{len(df)} transcripts found across {df['ticker'].nunique()} tickers")
    print(df.groupby("ticker").size().reindex(TICKERS, fill_value=0).to_string())


if __name__ == "__main__":
    main()
