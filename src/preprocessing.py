"""
preprocessing.py
----------------
Task 1 - Data Preparation.

Steps performed:
 1. Read the CSV file
 2. Display the dataset
 3. Check missing values
 4. Check duplicate records
 5. Remove duplicates
 6. Handle missing values
 7. Clean the ticket text
 8. Display basic information

`clean_text` is also imported by train.py and predict.py so that training
and prediction use EXACTLY the same cleaning.

Run on its own:  python src/preprocessing.py
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "tickets.csv"
CLEAN_CSV = ROOT / "data" / "tickets_clean.csv"


def clean_text(text: str) -> str:
    """Lowercase -> remove punctuation/symbols/digits -> collapse spaces."""
    text = str(text).lower()                    # 'Login' and 'login' become the same word
    text = text.replace("'", "")                # "week's" -> "weeks" (not "week s")
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()    # remove extra spaces
    return text


def load_and_clean(csv_path: Path = RAW_CSV, verbose: bool = True) -> pd.DataFrame:
    # 1 + 2. Read and display
    df = pd.read_csv(csv_path)
    if verbose:
        print("=" * 70)
        print("1. RAW DATASET (first 10 rows)")
        print("=" * 70)
        print(df.head(10).to_string(index=False))
        print(f"\nShape (rows, columns): {df.shape}")

    # 3. Missing values
    if verbose:
        print("\n" + "=" * 70)
        print("2. MISSING VALUES PER COLUMN")
        print("=" * 70)
        print(df.isnull().sum())

    # 4. Duplicates (ignore the ticket_id because each row has a unique id)
    cols_without_id = [c for c in df.columns if c != "ticket_id"]
    n_dup = df.duplicated(subset=cols_without_id).sum()
    if verbose:
        print(f"\nDuplicate records (ignoring ticket_id): {n_dup}")

    # 5. Remove exact duplicates
    df = df.drop_duplicates(subset=cols_without_id).copy()

    # 6. Handle missing values
    #    - description or category missing -> row is useless for training -> drop it
    #    - priority / status missing       -> fill with a sensible default
    before = len(df)
    df = df.dropna(subset=["ticket_description", "category"]).copy()
    dropped_missing = before - len(df)
    df["priority"] = df["priority"].fillna("Medium")
    df["status"] = df["status"].fillna("Open")

    # 7. Text cleaning
    df["clean_description"] = df["ticket_description"].apply(clean_text)
    df = df[df["clean_description"].str.len() > 0].copy()

    # Same sentence written twice with different punctuation/case -> duplicate after cleaning.
    # Remove them, otherwise the same text could land in BOTH train and test sets.
    before = len(df)
    df = df.drop_duplicates(subset=["clean_description"]).copy()
    dropped_text_dups = before - len(df)

    df = df.reset_index(drop=True)

    if verbose:
        print(f"Rows dropped because description/category was missing: {dropped_missing}")
        print(f"Rows dropped because cleaned text was duplicated:      {dropped_text_dups}")
        print("\n" + "=" * 70)
        print("3. AFTER CLEANING (first 10 rows)")
        print("=" * 70)
        print(df[["ticket_id", "ticket_description", "clean_description", "category"]].head(10).to_string(index=False))

        # 8. Basic information
        print("\n" + "=" * 70)
        print("4. BASIC INFORMATION")
        print("=" * 70)
        df.info()
        print("\nMissing values after cleaning:")
        print(df.isnull().sum().sum(), "missing values")
        print(f"Final number of records: {len(df)}")
        print(f"Number of categories:    {df['category'].nunique()}")

    return df


if __name__ == "__main__":
    cleaned = load_and_clean()
    cleaned.to_csv(CLEAN_CSV, index=False)
    print(f"\nSaved cleaned data to {CLEAN_CSV}")
