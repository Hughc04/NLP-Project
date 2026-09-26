"""Join transcripts to prices and compute post-event returns -> data/processed/events.csv.

Look-ahead rule. Day 0 is the last close at or before the moment the call's content
became public, and returns are measured forward from it:
  * before the open (< 09:30 ET) on a trading day -> day 0 = the PREVIOUS close,
    so day 1 is the call day itself (the market's first reaction)
  * intraday or after the close -> day 0 = the call day's close, so day 1 is the
    next trading day. For intraday calls this skips the partial same-day
    reaction rather than mixing pre-call and post-call moves.
ret_kd = close[day0 + k] / close[day0] - 1, and abn_kd subtracts the same-window QQQ return.
"""
import pandas as pd

from config import BENCHMARK, PROCESSED, RAW, TICKERS

HORIZONS = (1, 3, 5)


def classify_timing(call_time):
    minutes = call_time.hour * 60 + call_time.minute
    if minutes < 9 * 60 + 30:
        return "pre_market"
    if minutes < 16 * 60:
        return "intraday"
    return "after_close"


def day0_index(dates, call_time):
    """Position in `dates` of the day-0 close, per the look-ahead rule."""
    call_date = pd.Timestamp(call_time.date())
    idx = dates.searchsorted(call_date, side="right") - 1  # last trading day <= call date
    if idx < 0:
        return None
    on_trading_day = dates[idx] == call_date
    if classify_timing(call_time) == "pre_market" and on_trading_day:
        idx -= 1
    return idx if idx >= 0 else None


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(RAW / "prices.csv", index_col="date", parse_dates=True).sort_index()
    dates = prices.index
    transcripts = pd.read_csv(RAW / "transcripts.csv")

    rows = []
    for r in transcripts.itertuples():
        call_time = pd.Timestamp(r.call_time).tz_convert("America/New_York")
        i0 = day0_index(dates, call_time)
        row = {"date": call_time.date().isoformat(), "call_time": call_time.isoformat(),
               "timing": classify_timing(call_time), "ticker": r.ticker,
               "day0_date": None if i0 is None else dates[i0].date().isoformat()}
        for k in HORIZONS:
            if i0 is None or i0 + k >= len(dates):
                row[f"ret_{k}d"] = row[f"abn_{k}d"] = float("nan")
                continue
            stock = prices[r.ticker].iloc[i0 + k] / prices[r.ticker].iloc[i0] - 1
            market = prices[BENCHMARK].iloc[i0 + k] / prices[BENCHMARK].iloc[i0] - 1
            row[f"ret_{k}d"], row[f"abn_{k}d"] = stock, stock - market
        row["n_words"], row["text"] = r.n_words, r.text
        rows.append(row)

    events = pd.DataFrame(rows).sort_values(["date", "ticker"])
    events.to_csv(PROCESSED / "events.csv", index=False)

    ret_cols = [c for c in events.columns if c.startswith(("ret_", "abn_"))]
    print(f"{len(events)} events, {events['ticker'].nunique()} tickers, "
          f"{events['date'].min()} to {events['date'].max()}")
    print("\nTiming:", events["timing"].value_counts().to_dict())
    print("\nMissing values:", events[ret_cols].isna().sum().to_dict())
    print("\nEvents per ticker:", events.groupby("ticker").size().reindex(TICKERS, fill_value=0).to_dict())
    print("\nSpot check (first event per timing type):")
    for _, e in events.dropna(subset=["ret_1d"]).groupby("timing").head(1).iterrows():
        i0 = dates.searchsorted(pd.Timestamp(e["day0_date"]))
        print(f"  {e['ticker']} call {e['call_time']} [{e['timing']}]: day0={e['day0_date']} "
              f"close={prices[e['ticker']].iloc[i0]:.2f}, day1={dates[i0 + 1].date()} "
              f"close={prices[e['ticker']].iloc[i0 + 1]:.2f}, ret_1d={e['ret_1d']:+.4f}")


if __name__ == "__main__":
    main()
