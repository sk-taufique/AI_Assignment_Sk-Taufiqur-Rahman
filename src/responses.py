"""
responses.py
------------
OPTIONAL BONUS - Suggested customer response for each predicted category.

Approach: simple rule-based templates (no external API, no API keys needed).
"""

RESPONSES = {
    "Login Issue": (
        'Please use the "Forgot Password" option on the login page to reset your '
        "password. If the problem continues, please contact the support team."
    ),
    "Application Error": (
        "We are sorry for the trouble. Please refresh the page and try again. If the "
        "error continues, share a screenshot of the error message and the steps you "
        "followed so our technical team can investigate."
    ),
    "Report": (
        "Thank you for your request. Our team will prepare the report and share it with "
        "you shortly. Please mention the date range and format (PDF/Excel) you need."
    ),
    "Account Update": (
        "You can update your details under Profile > Account Settings. For security "
        "reasons, some changes (such as mobile number) need verification with an OTP."
    ),
    "Performance": (
        "We are sorry for the slowness. Please clear your browser cache or restart the "
        "application. Our team is checking system performance and will update you soon."
    ),
}

DEFAULT_RESPONSE = (
    "Thank you for contacting us. Our support team will review your request and "
    "get back to you shortly."
)


def suggest_response(category: str) -> str:
    return RESPONSES.get(category, DEFAULT_RESPONSE)
