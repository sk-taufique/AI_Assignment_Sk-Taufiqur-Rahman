"""
predict.py
----------
Task 5 + Task 6 (Option A) - Predict the category of a NEW ticket.

Run (interactive command-line app):   python src/predict.py
Run the 8 documented test cases:      python src/predict.py --test
Or predict a single text directly:    python src/predict.py "The app is very slow"
"""
import sys
from pathlib import Path

import joblib

from preprocessing import clean_text
from responses import suggest_response

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "trained_model.joblib"

# If the model is less sure than this, we warn the user.
LOW_CONFIDENCE = 0.50

_model = None


def load_model():
    """Load the saved model once and reuse it."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            sys.exit("Model not found. Please run:  python src/train.py")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_ticket(description: str) -> dict:
    """Return the predicted category, confidence (0-100) and a suggested reply."""
    model = load_model()
    cleaned = clean_text(description)          # SAME cleaning as in training
    probabilities = model.predict_proba([cleaned])[0]
    best = probabilities.argmax()
    category = model.classes_[best]
    confidence = float(probabilities[best]) * 100
    return {
        "category": category,
        "confidence": round(confidence, 1),
        "low_confidence": probabilities[best] < LOW_CONFIDENCE,
        "suggested_response": suggest_response(category),
    }


# Test tickets: wording is DIFFERENT from every row in the training dataset.
TEST_CASES = [
    ("I am unable to login because my password is not working.", "Login Issue"),
    ("Why does the system say my credentials are wrong?", "Login Issue"),
    ("An unexpected error popped up and the data did not get stored.", "Application Error"),
    ("Please share the sales summary for October.", "Report"),
    ("Kindly change the mobile number linked to my account.", "Account Update"),
    ("The website is taking ages to open.", "Performance"),
    ("The software crashed when I pressed the submit button.", "Application Error"),
    ("I want to get a download of all revenue figures for the last quarter.", "Report"),
]


def run_test_cases():
    print(f"{'Test Input':<72}{'Expected':<20}{'Predicted':<20}{'Conf.':<8}Result")
    print("-" * 130)
    correct = 0
    for text, expected in TEST_CASES:
        result = predict_ticket(text)
        ok = result["category"] == expected
        correct += ok
        print(f"{text:<72}{expected:<20}{result['category']:<20}{result['confidence']:<8}{'PASS' if ok else 'FAIL'}")
    print("-" * 130)
    print(f"Correct: {correct}/{len(TEST_CASES)}")


def show(description: str):
    result = predict_ticket(description)
    print(f"\nPredicted Category: {result['category']}")
    print(f"Confidence: {result['confidence']}%")
    if result["low_confidence"]:
        print("(Low confidence - please review this ticket manually.)")
    print(f"Suggested Response: {result['suggested_response']}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test_cases()
        return
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return

    print("=" * 40)
    print(" Customer Support Classifier")
    print("=" * 40)
    print("Type a ticket description (or 'q' to quit).")
    while True:
        text = input("\nEnter Ticket Description:\n> ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            print("Goodbye!")
            break
        if not text:
            print("Please type something.")
            continue
        show(text)


if __name__ == "__main__":
    main()
