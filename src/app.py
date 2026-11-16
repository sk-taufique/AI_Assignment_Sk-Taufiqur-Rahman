"""
app.py
------
Task 6 (Option B) - Simple web interface built with Flask.

Run:  python src/app.py     then open  http://127.0.0.1:5000
"""
from pathlib import Path

from flask import Flask, render_template, request

from predict import predict_ticket

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, template_folder=str(ROOT / "templates"))


@app.route("/", methods=["GET", "POST"])
def index():
    description, result, error = "", None, None
    if request.method == "POST":
        description = request.form.get("description", "").strip()
        if description:
            result = predict_ticket(description)
        else:
            error = "Please enter a ticket description."
    return render_template("index.html", description=description, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=False)
