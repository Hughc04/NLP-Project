import yfinance as yf

from config import BENCHMARK, END, RAW, START, TICKERS


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    symbols = TICKERS + [BENCHMARK]
    data = yf.download(symbols, start=START, end=END, auto_adjust=True,
                       progress=False)["Close"]
    data = data.dropna(how="all")
    data.index.name = "date"
    data.to_csv(RAW / "prices.csv")
    print(f"{data.shape[0]} trading days x {data.shape[1]} symbols "
          f"({data.index.min().date()} to {data.index.max().date()})")
    missing = data.isna().sum()
    gaps = missing[missing > 0]
    print("Missing values per symbol:", gaps.to_dict() if len(gaps) else "none")


if __name__ == "__main__":
    main()
