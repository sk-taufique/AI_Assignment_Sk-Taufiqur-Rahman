"""
generate_dataset.py
-------------------
Creates a synthetic customer-support ticket dataset (data/tickets.csv).

The data is written by hand (templates + natural variations), NOT downloaded
from the internet. A few duplicate rows and missing values are added on purpose
so that the data-cleaning step in preprocessing.py has something real to do.

Run:  python src/generate_dataset.py
"""
import random
from pathlib import Path

import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------------
# Hand-written ticket descriptions for each category
# ---------------------------------------------------------------------------
TICKETS = {
    "Login Issue": [
        "I forgot my password and cannot login",
        "Password reset is not working",
        "I cannot login to the application",
        "My account is locked after too many login attempts",
        "The reset password email never arrives in my inbox",
        "Unable to sign in with my username and password",
        "Login page keeps saying invalid credentials",
        "I am not receiving the OTP to log in",
        "Two factor authentication code is not accepted",
        "My login does not work after the password change",
        "I am locked out of my account",
        "Cannot access my account because I forgot the username",
        "Sign in button does nothing when I enter my password",
        "Forgot password link is broken",
        "The system says my password has expired and I cannot reset it",
        "Unable to log in using single sign on",
        "I keep getting logged out immediately after signing in",
        "Verification code for login is not coming to my phone",
        "Incorrect password error even though I typed the right one",
        "Please unlock my user account so I can log in",
        "I cannot sign in to the portal from my laptop",
        "My password is not working and I need to reset it",
        "Login failed for my account since this morning",
        "I did not get the password reset link",
    ],
    "Application Error": [
        "The application is showing an error while saving data",
        "Application gives error while saving",
        "I get an error message when I click submit",
        "The app crashes whenever I open the dashboard",
        "Error 500 internal server error appears on the page",
        "Something went wrong message is displayed when I upload a file",
        "The application throws an exception when I update a record",
        "Page shows a blank screen with an error code",
        "Unexpected error occurred while processing my request",
        "The system crashed and I lost my unsaved work",
        "I see a database error when I try to save the form",
        "The application freezes and then closes with an error",
        "Getting error 404 page not found on the orders screen",
        "Save button gives a failed to save error",
        "The app keeps crashing on startup",
        "Error popup appears every time I delete an item",
        "Bug in the application, data is not saved and an error is shown",
        "Upload fails with an unknown error message",
        "Application stopped working and shows a fatal error",
        "I get a timeout error when I submit the form",
        "The checkout screen displays an error and does not proceed",
        "There is a bug, the app closes unexpectedly when I press save",
        "An error is displayed when I try to edit my records",
        "Application error code 503 service unavailable",
    ],
    "Report": [
        "Please help me generate my monthly sales report",
        "Need monthly sales report",
        "I want to download the quarterly revenue report",
        "How can I export the report to Excel",
        "The report is missing data for last week",
        "Please send me the annual performance report",
        "I need a summary report of all customer orders",
        "Can you generate an inventory report for this month",
        "The sales report shows incorrect totals",
        "Unable to export report to PDF",
        "I would like a custom report of tickets by region",
        "Please schedule a weekly report to be emailed to me",
        "Need the expense report for the finance team",
        "Report does not include the filter I selected",
        "Where can I find the daily activity report",
        "Please create a report showing the sales by product",
        "I need last year's profit and loss report",
        "Request for a detailed usage report for my team",
        "The downloaded report is empty",
        "Could you share the customer growth report for this quarter",
        "I need help to build a dashboard report of monthly targets",
        "Need an attendance report for all employees",
        "Generate the monthly invoice report for me",
        "Please provide a report of last month's transactions",
    ],
    "Account Update": [
        "I need to change my registered mobile number",
        "Please update my email address on my account",
        "I want to change the name on my profile",
        "How do I update my billing address",
        "Please change my phone number in the system",
        "I need to update my company details",
        "Kindly modify the registered address of my account",
        "I want to change my account email to a new one",
        "Please update my profile picture",
        "Need to correct the spelling of my name in my account",
        "I want to add a new contact number to my profile",
        "Please change the registered email id",
        "I would like to update my bank details on my account",
        "Can you update my shipping address",
        "I need to change my notification preferences in my profile",
        "Please edit my date of birth in my account details",
        "I moved to a new city and need to update my address",
        "I want to close my old mobile number and register a new one",
        "Update the designation in my account profile",
        "Need to change the primary contact person on my account",
        "I would like to change my subscription plan details",
        "Please replace my old email with my new email address",
        "Request to update my account information",
        "I want to modify my user profile details",
    ],
    "Performance": [
        "The application is very slow today",
        "Application is very slow",
        "The system is taking too long to load",
        "Pages are loading very slowly since this morning",
        "The app is lagging and keeps freezing for a few seconds",
        "It takes more than a minute to open the dashboard",
        "Search is extremely slow and takes forever",
        "The website response time is too high",
        "The application became slow after the latest update",
        "Everything is running very slowly on my computer",
        "Uploading a file takes forever",
        "The screen is lagging when I scroll through the list",
        "Performance has degraded a lot in the last few days",
        "System is slow during peak hours",
        "The app is taking too long to respond",
        "Loading time of the home page is very high",
        "Application hangs for a long time when I open reports page",
        "Sluggish performance when many users are logged in",
        "The portal is so slow that I cannot finish my work",
        "Very long delay when I click on any button",
        "The mobile app is slow and laggy",
        "Page takes too much time to load",
        "System speed is very poor today",
        "Why is the application running so slow",
    ],
}

# Harder, more "real-life" tickets: shorter, vaguer, or sharing words with other
# categories (e.g. "slow" + "error", "report" + "error"). They make the task more
# realistic so the evaluation is not unrealistically perfect.
HARD_TICKETS = {
    "Login Issue": [
        "Cannot get in",
        "Access denied when I enter my credentials",
        "Why am I always asked to verify again before I can use my account",
        "Error message on the sign in screen",
        "Not able to enter the system since yesterday",
        "My id is blocked",
        "Login takes very long and then fails",
    ],
    "Application Error": [
        "Something is not working",
        "Report page shows an error",
        "The app is slow and then shows an error",
        "Data got lost after the system failed",
        "Error when changing my profile",
        "Exception while opening the screen",
        "Problem with the app, it does not respond and shows a message",
    ],
    "Report": [
        "Need data for last month",
        "I need numbers for my manager by Friday",
        "Is there a way to see the total sales per region",
        "Statistics for this quarter please",
        "Need an export of all my records",
        "Report is slow to download",
        "Want a summary of my team's work",
    ],
    "Account Update": [
        "Change my details",
        "Need to fix my information in the system",
        "My number has changed",
        "Wrong name is shown on my account",
        "Update required for my contact details, login works fine",
        "Please correct my address, the report shows the old one",
        "New phone, please change",
    ],
    "Performance": [
        "Too slow",
        "Waiting a long time for every page",
        "Application is slow when saving data",
        "The error appears only because the system is so slow",
        "Report takes ages to load",
        "System not responding quickly",
        "It was fast last week but not now",
    ],
}

PREFIXES = ["", "", "", "Hi, ", "Hello team, ", "Urgent: ", "Please note: ", "Dear support, "]
SUFFIXES = ["", "", "", ".", " Please help.", " Kindly resolve this.", " Thanks.", " as soon as possible."]

FIRST = ["Aarav", "Priya", "Rahul", "Sneha", "Vikram", "Ananya", "Rohan", "Kavya",
         "Arjun", "Neha", "Imran", "Pooja", "Karan", "Divya", "Sameer", "Meera",
         "Faisal", "Isha", "Nikhil", "Tanvi"]
LAST = ["Sharma", "Verma", "Khan", "Patel", "Reddy", "Das", "Iyer", "Singh", "Roy", "Nair"]

# Priority is *loosely* related to category, but with randomness (as in real life)
PRIORITY_WEIGHTS = {
    "Login Issue": [0.6, 0.3, 0.1],
    "Application Error": [0.6, 0.3, 0.1],
    "Report": [0.1, 0.5, 0.4],
    "Account Update": [0.1, 0.4, 0.5],
    "Performance": [0.5, 0.4, 0.1],
}
PRIORITIES = ["High", "Medium", "Low"]


def build_dataset() -> pd.DataFrame:
    rows = []
    for category, sentences in TICKETS.items():
        for sentence in sentences:
            # 1st copy: the sentence exactly as written
            # 2nd copy: the sentence with a natural prefix/suffix
            for variant in range(2):
                text = sentence
                if variant == 1:
                    text = random.choice(PREFIXES) + sentence + random.choice(SUFFIXES)
                    if text == sentence:  # force at least a small change
                        text = sentence + " Please help."
                rows.append({
                    "customer_name": f"{random.choice(FIRST)} {random.choice(LAST)}",
                    "ticket_description": text,
                    "date": f"2026-{random.randint(1, 10):02d}-{random.randint(1, 28):02d}",
                    "category": category,
                    "priority": random.choices(PRIORITIES, PRIORITY_WEIGHTS[category])[0],
                    "status": random.choices(["Open", "Closed", "In Progress"], [0.4, 0.45, 0.15])[0],
                })

    for category, sentences in HARD_TICKETS.items():
        for sentence in sentences:
            rows.append({
                "customer_name": f"{random.choice(FIRST)} {random.choice(LAST)}",
                "ticket_description": sentence,
                "date": f"2026-{random.randint(1, 10):02d}-{random.randint(1, 28):02d}",
                "category": category,
                "priority": random.choices(PRIORITIES, PRIORITY_WEIGHTS[category])[0],
                "status": random.choices(["Open", "Closed", "In Progress"], [0.4, 0.45, 0.15])[0],
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

    # -- Deliberately add a few DIRTY rows so the cleaning step is meaningful --
    dup = df.sample(5, random_state=1)               # 5 exact duplicates (except ID, added below)
    df = pd.concat([df, dup], ignore_index=True)
    df.loc[3, "priority"] = None                      # missing priority
    df.loc[17, "status"] = None                       # missing status
    df.loc[40, "ticket_description"] = None           # missing description (unusable)
    df.loc[60, "ticket_description"] = "   " + df.loc[60, "ticket_description"].upper() + "!!!  "  # messy text

    df.insert(0, "ticket_id", [f"T{i:03d}" for i in range(1, len(df) + 1)])
    # keep the columns in the order requested by the assignment (+ the extra ones)
    return df[["ticket_id", "customer_name", "ticket_description", "date",
               "priority", "status", "category"]]


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"
    out.parent.mkdir(exist_ok=True)
    data = build_dataset()
    data.to_csv(out, index=False)
    print(f"Saved {len(data)} rows to {out}")
    print(data["category"].value_counts())
