# Does earnings-call sentiment predict tech stock returns?

## Summary

I scored 133 earnings-call transcripts from 12 large tech companies (Nov 2022 to Sep 2026) with the Loughran-McDonald finance lexicon and tested whether the tone predicts the stock's return after the call.

- **Raw tone shows no relationship** with returns at any horizon (1-day correlation r = 0.01, p = 0.91).
- **Tone relative to the company's own history shows a weak positive link.** Spearman correlations are 0.20 to 0.26 across six return measures, and an OLS regression gives about +1.7 percentage points of next-day return per standard deviation of tone (p = 0.026, R² = 3%). This is suggestive, not established: it does not survive a correction for the many comparisons made, and the events are not independent (see Limitations).
- **An up/down classifier did not beat a majority-class guess** on held-out later events (54% accuracy against a 57% baseline, on only 35 test events).

The honest conclusion is that this sample does not demonstrate that lexicon sentiment predicts returns. It is consistent with markets already pricing in what management says, and it is also too small to rule out a modest effect.

This is a small research exercise, not investment advice.

## Data

| Item | Detail |
|---|---|
| Companies | AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, ORCL, ADBE, CRM, INTC, AMD |
| Text | Motley Fool earnings-call transcripts, found through the site's monthly sitemaps and scraped politely (3 second delay, cached) |
| Prices | Daily adjusted closes from yfinance, plus QQQ as a market benchmark |
| Sample | 133 events dated 2022-11-30 to 2026-09-10 (median transcript about 9,300 words) |
| Per company | AMD 12, AAPL 7, ADBE 8, AMZN 13, CRM 12, GOOGL 10, INTC 14, META 6, MSFT 12, NVDA 12, ORCL 14, TSLA 13 |

Coverage is uneven. Motley Fool had far fewer Apple (7), Meta (6) and Adobe (8) transcripts than the roughly 14 quarters available, so those companies carry less weight. The sitemaps listed 136 transcripts; 3 were dropped because the site's own call time was implausible (for example "2 a.m. ET"), since a wrong time could put the return window after the real call.

**Description of the outcome.** The average 1-day return after a call was -0.5% with a standard deviation of 9.4%, and only 42.9% of calls were followed by a rise.

## Method

**Event timing (the look-ahead rule).** All 133 calls took place after the market close. Day 0 is that day's close, and returns run forward from it: `ret_kd = close[day0 + k] / close[day0] - 1` for k = 1, 3, 5. The market-adjusted version (`abn_kd`) subtracts the same-window QQQ return. The call time comes from the header inside each transcript, not the page's publish timestamp, which is hours later. The code also handles pre-market calls, but none occurred in this sample, so that branch is untested.

**Sentiment.** Words are counted against the Loughran-McDonald master dictionary. The main score is `polarity = (positive - negative) / (positive + negative)`. Also computed: length-normalised tone, rates of negative, positive, uncertainty, litigious, strong-modal, weak-modal and constraining words. There is no negation handling and no stemming, so this is a baseline.

**Relative tone.** Raw tone differs a lot between companies (Adobe sounds upbeat on almost every call). The `_rel` features subtract a company's mean over its *earlier* calls only, so no future information is used. Each company's first call has no history and is dropped from those tests, leaving 121 events.

**Tests.**
- Pearson and Spearman correlations of every feature against six return measures (108 pairs), with p-values and a Benjamini-Hochberg correction.
- OLS regressions of return on polarity, with heteroskedasticity-robust (HC3) standard errors.
- Logistic regression predicting whether the return is above zero, with standardised features. The split is by date: the first 86 events (through 2025-10-29) train and the last 35 (from 2025-10-30) test, with no date shared. Metrics are accuracy with a Wilson confidence interval, precision, recall and F1 for the "up" class, and a one-sided binomial test against always guessing the training majority class.

**Look-ahead checks** are asserted in `src/analysis.py` and passed: returns start on or before the call date, history features use earlier calls only, train dates precede test dates, the scaler is fit on training rows only, and the word lists were never tuned on returns.

## Results

### Correlation and regression

| Sentiment measure | Return | Pearson r (p) | Spearman rho (p) |
|---|---|---|---|
| Raw polarity | 1-day | +0.01 (0.91) | +0.04 (0.66) |
| Raw polarity | 5-day | -0.01 (0.88) | +0.02 (0.86) |
| Relative polarity | 1-day | +0.18 (0.050) | +0.22 (0.014) |
| Relative polarity | 3-day | +0.19 (0.043) | +0.25 (0.005) |
| Relative polarity | 5-day | +0.18 (0.052) | +0.26 (0.004) |
| Relative polarity | 1-day, market-adjusted | +0.17 (0.064) | +0.22 (0.016) |

OLS of return on relative polarity (n = 121): coefficient 0.143 for the 1-day return (p = 0.026, R² = 0.032), 0.127 for the market-adjusted 1-day return (p = 0.038), and 0.175 for the 5-day return (p = 0.035). One standard deviation of relative polarity is 0.12, so the 1-day effect is about 1.7 percentage points, against a return standard deviation of 9.4 points. Raw polarity shows no effect (1-day coefficient 0.005, p = 0.90).

**Multiple comparisons.** Across all 108 feature-and-return pairs, 4 Pearson and 10 Spearman tests reached p < 0.05, against about 5 expected by chance for each. After Benjamini-Hochberg correction none were significant. The 10 Spearman hits come from two near-duplicate features (relative polarity and relative net tone), so they represent one finding, not ten.

### Up/down classification (35 test events, 43% of which rose)

| Target | Features | Accuracy (95% CI) | Majority baseline | Precision | Recall | F1 | p vs baseline |
|---|---|---|---|---|---|---|---|
| Return > 0 | Relative | 54% (38-70%) | 57% | 0.44 | 0.27 | 0.33 | 0.70 |
| Return > 0 | Raw | 51% (36-67%) | 57% | 0.40 | 0.27 | 0.32 | 0.80 |
| Market-adjusted > 0 | Relative | 63% (46-77%) | 57% | 1.00 | 0.13 | 0.24 | 0.31 |
| Market-adjusted > 0 | Raw | 57% (41-72%) | 57% | 0.50 | 0.20 | 0.29 | 0.57 |

No model is significantly better than the baseline. The best-looking row catches only 13% of the actual rises. Its coefficients (`classifier_coefficients.png`) should not be read one by one: several features overlap heavily, and for instance the negative-word rate gets a positive weight even though its own correlation with returns is negative.

Figures: `scatter_sentiment_vs_return.png`, `correlation_heatmap.png`, `classifier_coefficients.png`. Full tables: `correlations.csv`, `regressions.csv`, `classification.csv`.

## Limitations

- **Small sample.** 121 to 133 events, and only 35 in the classifier's test set, so the accuracy interval spans about 30 points.
- **Events are not independent.** Tech stocks report in the same weeks and move together with the market and each other; 27 dates had more than one call. The 3-day and 5-day windows also overlap other events. Standard p-values are therefore probably too optimistic. Clustering by date or quarter would be a better test.
- **Researcher choices.** I built the relative-tone measure after seeing that some companies always sound upbeat, and I looked at many features and horizons. Even with a correction it is the only thing that hints at a signal, so it needs confirming on new data.
- **Efficient markets.** Much of what management says is already priced in through the earnings release earlier that day, and the market reacts to *surprise* (results versus expectations), guidance and the numbers themselves. This project did not control for any of them, so tone may just be a weak proxy for good or bad news.
- **Lexicon limits.** The Loughran-McDonald lists were built for annual reports, not spoken calls. Prepared remarks and Q&A are scored together, and there is no negation handling ("not good" counts as positive).
- **Data quality.** Scraped pages had inconsistent headers (fixed by supporting three formats) and a few wrong call times (dropped). Coverage gaps mean some companies are under-represented. Motley Fool's terms of service may restrict scraping, so the raw transcripts are not published in this repo.
- **Selection.** The 12 companies are today's well-known tech leaders, chosen with hindsight, in one sector and one period, so the results may not carry over to other stocks, sectors or market conditions.
- **Not covered.** Pre-market calls, transaction costs, and any trading strategy.

## Not done yet

- **Phase 3 (deferred by choice):** FinBERT scores and separate hedging-language features.
- Useful next steps: control for earnings surprise, use more events (other sectors or sources), test with date-clustered standard errors or a permutation test, and evaluate on a fresh time period.

## Reproducing

See `README.md`. The scripts run in order; everything except the raw data in `data/raw/` and the full-text `data/processed/events.csv` is in the repository.
