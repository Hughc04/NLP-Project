"""Phase 4: does lexicon sentiment predict post-earnings-call returns?

Reads data/processed/scored.csv and writes tables + plots to data/results/.

Look-ahead safeguards (asserted in check_lookahead):
  * returns start at a close at or before the call (day0 <= call date)
  * "_rel" features subtract the company's mean over its PRIOR calls only, so no
    future information about a company's typical tone is used
  * train/test is a split by date: every training call precedes every test call
  * the scaler is fit inside the pipeline on training rows only
  * the LM word lists are fixed by the dictionary, never tuned on returns
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from config import PROCESSED, RESULTS

FEATURES = ["polarity", "net_tone", "neg_rate", "pos_rate", "uncertainty_rate",
            "litigious_rate", "modal_strong_rate", "modal_weak_rate", "constraining_rate"]
REL_FEATURES = [f"{f}_rel" for f in FEATURES]
RETURNS = ["ret_1d", "ret_3d", "ret_5d", "abn_1d", "abn_3d", "abn_5d"]
TRAIN_FRACTION = 0.7
SEED = 0

# Reference palette (light mode): single series blue, blue<->red diverging with a gray midpoint
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("blue_red", [RED, NEUTRAL, BLUE])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.family": ["Segoe UI", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def load():
    df = pd.read_csv(PROCESSED / "scored.csv", parse_dates=["date", "day0_date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    for f in FEATURES:
        prior_mean = df.groupby("ticker")[f].transform(lambda s: s.shift(1).expanding().mean())
        df[f"{f}_rel"] = df[f] - prior_mean  # NaN for a company's first call
    return df


def split_by_date(df):
    share = df.groupby("date").size().cumsum() / len(df)
    cutoff = share[share >= TRAIN_FRACTION].index[0]
    return df[df["date"] <= cutoff], df[df["date"] > cutoff]


def check_lookahead(df, valid, train, test):
    assert (df["day0_date"] <= df["date"]).all(), "returns start after the call date"
    first = df.groupby("ticker").head(1)
    assert first[REL_FEATURES].isna().all().all(), "a first call has a history-based feature"
    assert len(valid) == len(df) - df["ticker"].nunique(), "unexpected rows dropped"
    assert train["date"].max() < test["date"].min(), "train and test dates overlap"
    print("Look-ahead checks passed:")
    print("  - all day-0 closes are on or before the call date")
    print(f"  - history-based features are empty for each company's first call ({df['ticker'].nunique()} rows dropped)")
    print(f"  - train {train['date'].min():%Y-%m-%d} to {train['date'].max():%Y-%m-%d} ({len(train)} events) "
          f"strictly before test {test['date'].min():%Y-%m-%d} to {test['date'].max():%Y-%m-%d} ({len(test)} events)")
    print("  - scaler fit on train only (inside the pipeline); LM word lists fixed, not tuned on returns")


def correlations(df):
    rows = []
    for feat in FEATURES + REL_FEATURES:
        for ret in RETURNS:
            d = df[[feat, ret]].dropna()
            r, p = stats.pearsonr(d[feat], d[ret])
            rho, ps = stats.spearmanr(d[feat], d[ret])
            rows.append({"feature": feat, "ret": ret, "n": len(d), "pearson_r": r,
                         "pearson_p": p, "spearman_rho": rho, "spearman_p": ps})
    out = pd.DataFrame(rows)
    out["pearson_q"] = multipletests(out["pearson_p"], method="fdr_bh")[1]
    out["spearman_q"] = multipletests(out["spearman_p"], method="fdr_bh")[1]
    return out


def regressions(df):
    rows = []
    for x in ["polarity", "polarity_rel"]:
        for y in ["ret_1d", "abn_1d", "ret_5d", "abn_5d"]:
            d = df[[x, y]].dropna()
            fit = sm.OLS(d[y], sm.add_constant(d[x])).fit(cov_type="HC3")
            rows.append({"x": x, "y": y, "n": len(d), "coef": fit.params[x], "se": fit.bse[x],
                         "p": fit.pvalues[x], "r2": fit.rsquared})
    return pd.DataFrame(rows)


def evaluate(train, test, target, feats):
    ytr, yte = (train[target] > 0).astype(int), (test[target] > 0).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED))
    model.fit(train[feats], ytr)
    pred = model.predict(test[feats])
    base_pred = np.full(len(yte), int(ytr.mean() >= 0.5))  # always the training majority
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    hits, n = int((pred == yte).sum()), len(yte)
    lo, hi = proportion_confint(hits, n, method="wilson")
    base_acc = accuracy_score(yte, base_pred)
    row = {"target": f"{target} > 0", "features": "relative" if feats[0].endswith("_rel") else "raw",
           "n_train": len(train), "n_test": n, "test_up_share": yte.mean(),
           "accuracy": hits / n, "acc_ci_low": lo, "acc_ci_high": hi,
           "precision": prec, "recall": rec, "f1": f1,
           "baseline_acc": base_acc,
           "p_vs_baseline": stats.binomtest(hits, n, base_acc, alternative="greater").pvalue}
    return row, model


def style_title(ax, title):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)


def plot_scatter(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    panels = [("polarity", "Raw tone"), ("polarity_rel", "Tone relative to the company's own history")]
    for ax, (x, name) in zip(axes, panels):
        d = df[[x, "ret_1d"]].dropna()
        y = d["ret_1d"] * 100
        ax.scatter(d[x], y, s=38, color=BLUE, edgecolor=SURFACE, linewidth=0.8, alpha=0.9, zorder=3)
        slope, intercept, r, p, _ = stats.linregress(d[x], y)
        xs = np.linspace(d[x].min(), d[x].max(), 50)
        ax.plot(xs, slope * xs + intercept, color=INK2, lw=2, zorder=2)
        ax.axhline(0, color=AXIS, lw=1, zorder=1)
        style_title(ax, f"{name}\nr = {r:+.2f}, p = {p:.2f}, n = {len(d)}")
        ax.set_xlabel("LM polarity: (positive - negative) / (positive + negative)")
    axes[0].set_ylabel("1-day return after the call (%)")
    fig.tight_layout()
    fig.savefig(RESULTS / "scatter_sentiment_vs_return.png", dpi=160)
    plt.close(fig)


def plot_heatmap(corr):
    order = FEATURES + REL_FEATURES
    rho = corr.pivot(index="feature", columns="ret", values="spearman_rho").loc[order, RETURNS]
    pv = corr.pivot(index="feature", columns="ret", values="spearman_p").loc[order, RETURNS]
    lim = max(0.3, float(np.abs(rho.values).max()))
    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.imshow(rho.values, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(RETURNS)), RETURNS)
    ax.set_yticks(range(len(order)), order)
    for i in range(len(order)):
        for j in range(len(RETURNS)):
            star = "*" if pv.values[i, j] < 0.05 else ""
            ax.text(j, i, f"{rho.values[i, j]:+.2f}{star}", ha="center", va="center", fontsize=8.5, color=INK)
    ax.axhline(len(FEATURES) - 0.5, color=SURFACE, lw=4)  # gap between raw and relative features
    for side in ax.spines.values():
        side.set_visible(False)
    style_title(ax, "Spearman correlation of sentiment features with returns\n"
                    "(blue = positive, red = negative, * = p < 0.05 before multiple-testing correction)")
    fig.tight_layout()
    fig.savefig(RESULTS / "correlation_heatmap.png", dpi=160)
    plt.close(fig)


def plot_coefficients(model, feats):
    coefs = pd.Series(model.named_steps["logisticregression"].coef_[0], index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(coefs.index, coefs.values, color=[BLUE if c > 0 else RED for c in coefs], height=0.6)
    ax.axvline(0, color=AXIS, lw=1)
    for y, c in enumerate(coefs.values):
        ax.text(c + (0.015 if c >= 0 else -0.015), y, f"{c:+.2f}", va="center",
                ha="left" if c >= 0 else "right", fontsize=8.5, color=INK2)
    ax.grid(False)
    ax.set_xlim(coefs.min() - 0.12, coefs.max() + 0.12)  # room for the value labels
    ax.set_xlabel("Standardised logistic-regression coefficient (blue = pushes toward 'up')")
    style_title(ax, "Which features the classifier leans on (fit on training events only)\n"
                    "These features overlap heavily, so individual bars are unreliable")
    fig.tight_layout()
    fig.savefig(RESULTS / "classifier_coefficients.png", dpi=160)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load()
    valid = df.dropna(subset=REL_FEATURES)
    train, test = split_by_date(valid)
    check_lookahead(df, valid, train, test)

    corr = correlations(df)
    corr.to_csv(RESULTS / "correlations.csv", index=False)
    print("\n== Correlations: headline (polarity) ==")
    head = corr[corr["feature"].isin(["polarity", "polarity_rel"])]
    print(head[["feature", "ret", "n", "pearson_r", "pearson_p", "spearman_rho", "spearman_p"]]
          .round(3).to_string(index=False))
    n_tests = len(corr)
    print(f"\nAll {n_tests} feature x return pairs: nominal p<0.05 in "
          f"{(corr.pearson_p < .05).sum()} Pearson and {(corr.spearman_p < .05).sum()} Spearman tests "
          f"(about {0.05 * n_tests:.1f} expected by chance); "
          f"significant after Benjamini-Hochberg (q<0.10): "
          f"{(corr.pearson_q < .10).sum()} Pearson, {(corr.spearman_q < .10).sum()} Spearman")

    reg = regressions(df)
    reg.to_csv(RESULTS / "regressions.csv", index=False)
    print("\n== OLS: return = a + b * sentiment (HC3 robust SEs) ==")
    print(reg.round(4).to_string(index=False))

    print("\n== Up/down classification (logistic regression, time-based split) ==")
    results, primary = [], None
    for target in ["ret_1d", "abn_1d"]:
        for feats in (REL_FEATURES, FEATURES):
            row, model = evaluate(train, test, target, feats)
            results.append(row)
            if target == "ret_1d" and feats is REL_FEATURES:
                primary = model
    clf = pd.DataFrame(results)
    clf.to_csv(RESULTS / "classification.csv", index=False)
    print(clf.round(3).to_string(index=False))

    plot_scatter(df)
    plot_heatmap(corr)
    plot_coefficients(primary, REL_FEATURES)
    print(f"\nPlots and tables written to {RESULTS}")


if __name__ == "__main__":
    main()
