"""
train.py
--------
Task 3 + Task 4 - Build, train and evaluate the ML model.

Pipeline:
    ticket description -> clean_text -> TF-IDF -> Logistic Regression -> category

Run:  python src/train.py
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing import load_and_clean

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "trained_model.joblib"
SHOT_DIR = ROOT / "screenshots"
SHOT_DIR.mkdir(exist_ok=True)
MODEL_PATH.parent.mkdir(exist_ok=True)

RANDOM_STATE = 42


def make_pipeline(classifier):
    """TF-IDF turns text into numbers; the classifier learns from those numbers."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words=None, sublinear_tf=True)),
        ("clf", classifier),
    ])


def main():
    df = load_and_clean(verbose=False)
    X = df["clean_description"]
    y = df["category"]

    # ---- Task 4: 80 / 20 split (stratify keeps category proportions equal) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"Training samples: {len(X_train)}  |  Testing samples: {len(X_test)}\n")

    # ---- Compare 3 simple algorithms using 5-fold cross validation on TRAIN data ----
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=10),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(),
    }
    print("Model comparison (5-fold cross-validation accuracy on training data):")
    for name, clf in candidates.items():
        scores = cross_val_score(make_pipeline(clf), X_train, y_train, cv=5, scoring="accuracy")
        print(f"  {name:<20} mean = {scores.mean():.3f}  (+/- {scores.std():.3f})")

    # ---- Final model: Logistic Regression (gives probabilities -> confidence %) ----
    model = make_pipeline(candidates["Logistic Regression"])
    model.fit(X_train, y_train)

    # ---- Evaluate on the unseen test data ----
    y_pred = model.predict(X_test)
    labels = sorted(y.unique())

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 60)
    print("TEST RESULTS (Logistic Regression + TF-IDF)")
    print("=" * 60)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%  (weighted average)")
    print(f"Recall   : {rec * 100:.2f}%  (weighted average)")
    print(f"F1 Score : {f1 * 100:.2f}%  (weighted average)")
    print("\nPer-category report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(labels)
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted category")
    plt.ylabel("Actual category")
    plt.title("Confusion Matrix (test data)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(SHOT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # ---- Save model ----
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # ---- Show what the model learned: top words for each category ----
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    names = tfidf.get_feature_names_out()
    print("\nTop 5 words/phrases the model uses for each category:")
    for i, cat in enumerate(clf.classes_):
        top = clf.coef_[i].argsort()[-5:][::-1]
        print(f"  {cat:<18}: {', '.join(names[top])}")


if __name__ == "__main__":
    main()
