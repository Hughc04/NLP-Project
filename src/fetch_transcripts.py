"""Download the transcripts listed in data/raw/transcript_urls.csv and parse them.

Pages are cached in data/raw/transcripts/ (one fetch per URL, 3 second delay).
Writes data/raw/transcripts.csv with the call time (US/Eastern) and the body text.
The call time comes from the header in the transcript body, e.g.
"Q4 2022 Earnings Call Feb 02, 2023 , 5:30 p.m. ET". The page's own publish
timestamp is NOT used: it is the posting time, hours after the call.
Rows with no parseable call time are dropped and reported, never guessed.
"""
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import RAW

HEADERS = {"User-Agent": "Mozilla/5.0 (student research project; hughconboy@gmail.com)"}
PAGE_DIR = RAW / "transcripts"
DELAY_SECONDS = 3
MIN_CALL_HOUR, MAX_CALL_HOUR = 8, 19  # plausible US earnings-call window, ET

_TIME = r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\s*([A-Z]{2,4})?"
# Older pages: "Q4 2022 Earnings Call Feb 02, 2023 , 5:30 p.m. ET"
# Newer pages: "Date Wednesday, November 19, 2025 at 5 p.m. ET", also written
# "DATE ... Feb. 25, 2026 at 5 p.m. ET" and "Date Jan. 28, 2026, 4:30 p.m. ET"
CALL_TIME_PATTERNS = [
    re.compile(r"Earnings Call\s+([A-Z][a-z]{2,8})\.?\s+(\d{1,2})\s*,\s*(\d{4})\s*,?\s*" + _TIME),
    re.compile(r"\b(?i:date)\s+(?:[A-Z][a-z]+,\s+)?([A-Z][a-z]{2,8})\.?\s+(\d{1,2})\s*,\s*(\d{4})\s*,?\s*(?:at\s+)?" + _TIME),
]
TIMEZONES = {"ET": "America/New_York", "EST": "America/New_York", "EDT": "America/New_York",
             "CT": "America/Chicago", "CST": "America/Chicago", "CDT": "America/Chicago",
             "PT": "America/Los_Angeles", "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles",
             "UTC": "UTC", "GMT": "UTC"}


def fetch(url, slug):
    cache = PAGE_DIR / f"{slug}.html"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    cache.write_text(resp.text, encoding="utf-8")
    time.sleep(DELAY_SECONDS)
    return resp.text


def parse_call_time(text):
    """Return a US/Eastern-aware timestamp from the body header, or None."""
    for pattern in CALL_TIME_PATTERNS:
        m = pattern.search(text[:1500])
        if m:
            break
    else:
        return None
    month, day, year, hour, minute, ampm, zone = m.group(1, 2, 3, 4, 5, 6, 7)
    tz = TIMEZONES.get((zone or "ET").upper())
    if tz is None:
        return None  # unknown zone label: drop rather than guess
    try:
        naive = datetime.strptime(f"{month[:3]} {day} {year} {hour}:{minute or '00'} {ampm.upper()}M",
                                  "%b %d %Y %I:%M %p")
    except ValueError:
        return None
    return pd.Timestamp(naive).tz_localize(tz).tz_convert("America/New_York")


def parse_page(html):
    soup = BeautifulSoup(html, "lxml")
    body = soup.find(id="article-body-transcript")
    if body is None:
        return None, None
    text = " ".join(body.get_text(" ").split())
    call_time = parse_call_time(text)
    # Drop the "Image source / Contents" preamble; keep from the remarks onward.
    start = text.find("Prepared Remarks:")
    if start != -1:
        text = text[start + len("Prepared Remarks:"):].strip()
    return call_time, text


def main():
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    urls = pd.read_csv(RAW / "transcript_urls.csv")
    rows, dropped = [], []
    for i, r in enumerate(urls.itertuples(), 1):
        try:
            html = fetch(r.url, r.slug)
        except requests.RequestException as e:
            dropped.append((r.ticker, r.slug, f"fetch failed: {e}"))
            continue
        call_time, text = parse_page(html)
        if text is None:
            dropped.append((r.ticker, r.slug, "no transcript body found"))
        elif call_time is None:
            dropped.append((r.ticker, r.slug, "call time not found"))
        elif not MIN_CALL_HOUR <= call_time.hour < MAX_CALL_HOUR:
            # The site's header has typos (e.g. "2 a.m. ET"); a wrong time could put
            # day 0 after the real call, so drop instead of guessing a correction.
            dropped.append((r.ticker, r.slug, f"implausible call time {call_time:%H:%M} ET"))
        else:
            rows.append({"ticker": r.ticker, "url": r.url, "slug": r.slug,
                         "call_time": call_time.isoformat(), "n_words": len(text.split()),
                         "text": text})
        print(f"[{i}/{len(urls)}] {r.ticker} {r.slug}", flush=True)

    df = pd.DataFrame(rows).sort_values(["ticker", "call_time"])
    df.to_csv(RAW / "transcripts.csv", index=False)
    print(f"\nParsed {len(df)} of {len(urls)} transcripts")
    if dropped:
        print(f"Dropped {len(dropped)}:")
        for d in dropped:
            print("  ", *d)


if __name__ == "__main__":
    main()
