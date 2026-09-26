# NLP-Project

Does the tone of an earnings call predict the stock's move afterwards? This project scores earnings-call transcripts from 12 large tech companies with the Loughran-McDonald finance lexicon and tests the scores against post-call returns.

**Result in one line:** raw tone shows no relationship with returns; tone relative to a company's own history shows a weak link that does not survive multiple-testing correction, and an up/down classifier does not beat a majority-class guess. Details and caveats are in [`data/results/REPORT.md`](data/results/REPORT.md).

## Setup (Windows, Python 3.12)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Download the Loughran-McDonald master dictionary CSV from
[sraf.nd.edu](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) and save it as `data/raw/LM_MasterDictionary.csv`.

## Running the pipeline

Run from the project folder, in order:

| Step | Script | Writes |
|---|---|---|
| 1 | `src/fetch_prices.py` | `data/raw/prices.csv` (yfinance) |
| 2 | `src/discover_transcripts.py` | `data/raw/transcript_urls.csv` (from Motley Fool sitemaps) |
| 3 | `src/fetch_transcripts.py` | `data/raw/transcripts.csv` (about 7 minutes; pages are cached) |
| 4 | `src/build_events.py` | `data/processed/events.csv` (returns from the last close before each call) |
| 5 | `src/lexicon_sentiment.py` | `data/processed/scored.csv` |
| 6 | `src/analysis.py` | tables and plots in `data/results/` |

Each script is run like `.venv\Scripts\python src\fetch_prices.py`. Tickers and dates are set in `src/config.py`.

## What is and is not in the repo

Committed: the code, `data/processed/scored.csv` (scores and returns, no text) and everything in `data/results/`.

Not committed: `data/raw/` and `data/processed/events.csv`, because they contain full copyrighted transcript text. Steps 1 to 5 recreate them.

## Status

| Phase | State |
|---|---|
| 1. Data infrastructure | Done |
| 2. Lexicon sentiment baseline | Done |
| 3. FinBERT and hedging features | Deferred |
| 4. Correlation, regression, classifier, plots | Done |
| 5. Write-up | Done (`data/results/REPORT.md`) |
