"""
Built-in synthetic test pairs — one per comparison mode.
Used by GET /test to smoke-test the full pipeline.
"""

NEAR_DUPLICATE_TEXT = (
    "Invoice INV-2024-001\n"
    "Date: 2024-01-15\n"
    "Bill to: Acme Corporation\n"
    "Item: Consulting Services - Q4 2023\n"
    "Amount: $4,500.00\n"
    "Payment terms: NET-30\n"
    "Due date: 2024-02-14"
)

NEAR_DUPLICATE_IMAGE_TEXT = (
    "Invoice INV-2024-001\n"
    "Date: January 15, 2024\n"
    "Customer: Acme Corporation\n"
    "Description: Consulting Services Q4 2023\n"
    "Total: $4,500.00\n"
    "Terms: NET-30\n"
    "Due: February 14 2024"
)

PARAPHRASE_TICKET_A = (
    "Ticket TKT-555\n"
    "The authentication service is returning HTTP 500 errors on the login endpoint.\n"
    "All users are affected. Priority: HIGH.\n"
    "Assigned to: DevOps Team"
)

PARAPHRASE_TICKET_B = (
    "Ticket TKT-555\n"
    "Login system keeps crashing with internal server errors.\n"
    "Entire user base impacted. Urgent fix needed.\n"
    "Team: DevOps"
)

UNRELATED_DOC_1 = (
    "Invoice INV-9999\n"
    "Date: 2024-03-01\n"
    "Amount: $12,750.00\n"
    "Client: Beta Technologies Ltd"
)

UNRELATED_DOC_2 = (
    "Weather Forecast - London\n"
    "Saturday: Overcast with light rain in the morning.\n"
    "Sunday: Partly cloudy, high of 14 degrees Celsius.\n"
    "No travel disruptions expected."
)

SYNTHETIC_PAIRS = [
    {
        "label": "near_duplicate_invoice",
        "mode": "text-image",
        "doc1_text": NEAR_DUPLICATE_TEXT,
        "doc2_text": NEAR_DUPLICATE_IMAGE_TEXT,
        "expected_min": 0.55,
        "expected_max": 1.00,
    },
    {
        "label": "paraphrase_ticket",
        "mode": "image-image",
        "doc1_text": PARAPHRASE_TICKET_A,
        "doc2_text": PARAPHRASE_TICKET_B,
        "expected_min": 0.40,
        "expected_max": 0.85,
    },
    {
        "label": "unrelated_documents",
        "mode": "text-text",
        "doc1_text": UNRELATED_DOC_1,
        "doc2_text": UNRELATED_DOC_2,
        "expected_min": 0.00,
        "expected_max": 0.35,
    },
]
