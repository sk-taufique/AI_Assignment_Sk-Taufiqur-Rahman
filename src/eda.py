"""
eda.py
------
Task 2 - Exploratory Data Analysis.

Prints the counts and saves 3 charts into screenshots/.

Run:  python src/eda.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # lets the script save charts even without a display
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_and_clean

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "screenshots"
OUT.mkdir(exist_ok=True)


def main():
    df = load_and_clean(verbose=False)

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print(f"Total number of tickets: {len(df)}\n")
    print("Tickets per category:\n", df["category"].value_counts(), "\n")
    print("Tickets per priority:\n", df["priority"].value_counts(), "\n")
    print("Tickets per status:\n", df["status"].value_counts(), "\n")
    print("Category distribution (%):\n", (df["category"].value_counts(normalize=True) * 100).round(1), "\n")

    df["word_count"] = df["clean_description"].str.split().str.len()
    print("Average words per ticket:", round(df["word_count"].mean(), 1))
    print("Shortest / longest ticket (words):", df["word_count"].min(), "/", df["word_count"].max())

    sns.set_theme(style="whitegrid")

    # Chart 1: category distribution
    plt.figure(figsize=(8, 5))
    order = df["category"].value_counts().index
    ax = sns.countplot(data=df, y="category", order=order, hue="category", legend=False, palette="viridis")
    for c in ax.containers:
        ax.bar_label(c, padding=3)
    plt.title("Chart 1: Number of Tickets per Category")
    plt.xlabel("Number of tickets")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(OUT / "category_distribution.png", dpi=150)
    plt.close()

    # Chart 2: priority distribution
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x="priority", order=["High", "Medium", "Low"],
                       hue="priority", legend=False, palette="rocket")
    for c in ax.containers:
        ax.bar_label(c, padding=3)
    plt.title("Chart 2: Number of Tickets per Priority")
    plt.xlabel("Priority")
    plt.ylabel("Number of tickets")
    plt.tight_layout()
    plt.savefig(OUT / "priority_distribution.png", dpi=150)
    plt.close()

    # Chart 3: priority within each category (shows which categories are urgent)
    plt.figure(figsize=(9, 5))
    sns.countplot(data=df, x="category", hue="priority", hue_order=["High", "Medium", "Low"], palette="rocket")
    plt.title("Chart 3: Priority within each Category")
    plt.xlabel("")
    plt.ylabel("Number of tickets")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT / "priority_by_category.png", dpi=150)
    plt.close()

    print(f"\nCharts saved in: {OUT}")


if __name__ == "__main__":
    main()
