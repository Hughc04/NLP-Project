"""Score each transcript with the Loughran-McDonald finance lexicon -> data/processed/scored.csv.

Source: LM master dictionary (Notre Dame SRAF). In the CSV a category column holds
the year the word was added, or 0 if the word is not in that category.
Baseline only: no negation handling and no stemming (the dictionary lists inflected forms).
scored.csv leaves out the transcript text so it can be committed.
"""
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from config import PROCESSED, RAW

CATEGORIES = ["Negative", "Positive", "Uncertainty", "Litigious",
              "Strong_Modal", "Weak_Modal", "Constraining"]
tokenizer = RegexpTokenizer(r"[A-Za-z]+")


def load_lexicon():
    lm = pd.read_csv(RAW / "LM_MasterDictionary.csv", usecols=["Word"] + CATEGORIES)
    return {cat: set(lm.loc[lm[cat] > 0, "Word"]) for cat in CATEGORIES}


def score_text(text, lexicon):
    tokens = [t.upper() for t in tokenizer.tokenize(text)]
    counts = {cat: sum(t in words for t in tokens) for cat, words in lexicon.items()}
    n = len(tokens)
    pos, neg = counts["Positive"], counts["Negative"]
    return {
        "n_tokens": n,
        "pos": pos,
        "neg": neg,
        # (pos - neg) / (pos + neg): -1 all negative, +1 all positive
        "polarity": (pos - neg) / (pos + neg) if pos + neg else 0.0,
        # length-normalised alternative, per token
        "net_tone": (pos - neg) / n if n else 0.0,
        "neg_rate": neg / n if n else 0.0,
        "pos_rate": pos / n if n else 0.0,
        "uncertainty_rate": counts["Uncertainty"] / n if n else 0.0,
        "litigious_rate": counts["Litigious"] / n if n else 0.0,
        "modal_strong_rate": counts["Strong_Modal"] / n if n else 0.0,
        "modal_weak_rate": counts["Weak_Modal"] / n if n else 0.0,
        "constraining_rate": counts["Constraining"] / n if n else 0.0,
    }


def main():
    lexicon = load_lexicon()
    print({cat: len(words) for cat, words in lexicon.items()})
    events = pd.read_csv(PROCESSED / "events.csv")
    scores = pd.DataFrame([score_text(t, lexicon) for t in events["text"]])
    scored = pd.concat([events.drop(columns=["text", "n_words"]), scores], axis=1)
    scored.to_csv(PROCESSED / "scored.csv", index=False)

    print(f"\n{len(scored)} events scored")
    print(scored[["polarity", "net_tone", "neg_rate", "uncertainty_rate"]].describe().round(4).to_string())
    cols = ["ticker", "date", "polarity", "pos", "neg", "ret_1d"]
    print("\nMost negative:\n", scored.nsmallest(3, "polarity")[cols].to_string(index=False))
    print("\nMost positive:\n", scored.nlargest(3, "polarity")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
