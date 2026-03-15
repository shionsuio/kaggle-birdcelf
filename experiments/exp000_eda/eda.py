"""Quick tabular EDA for BirdCLEF 2026."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv")
    taxonomy = pd.read_csv(ROOT / "taxonomy.csv")
    soundscape_labels = pd.read_csv(ROOT / "train_soundscapes_labels.csv")
    sample_submission = pd.read_csv(ROOT / "sample_submission.csv")

    target_labels = sample_submission.columns[1:]

    print("train rows:", len(train))
    print("taxonomy rows:", len(taxonomy))
    print("soundscape label rows:", len(soundscape_labels))
    print("submission targets:", len(target_labels))
    print()

    print("top primary_label counts:")
    print(train["primary_label"].value_counts().head(10).to_string())
    print()

    print("metadata columns:")
    print(", ".join(train.columns))
    print()

    print("labels in train but not submission:", len(set(train["primary_label"]) - set(target_labels)))
    print("labels in submission but not train:", len(set(target_labels) - set(train["primary_label"])))


if __name__ == "__main__":
    main()
