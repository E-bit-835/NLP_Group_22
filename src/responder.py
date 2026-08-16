"""
Turns an NLU result into a final natural-language reply, always
grounded in the real CSV data (never invented).
"""
from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd

from . import data_access as da
from .model_loader import Resources, get_resources
from .nlu import NluResult, classify

GREETING_REPLIES = (
    "Hello! Welcome to Mobixa. I can help with phone prices, availability, "
    "comparisons, accessories, and repairs. What are you looking for today?"
)
GOODBYE_REPLIES = "Thanks for visiting Mobixa. Have a great day!"
THANKS_REPLIES = "You're welcome! Let me know if there's anything else I can help with."
UNKNOWN_REPLY = (
    "Sorry, I couldn't quite understand that. I can help with phone prices, "
    "availability, comparisons, accessories, repairs, and store FAQs — "
    "could you rephrase your question?"
)


def _parse_number(value: object) -> Optional[float]:
    if value is None:
        return None
    match = re.search(r"[\d,]+(?:\.\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _format_phone_line(row: pd.Series) -> str:
    return (
        f"{row['Company Name']} {row['Model Name']} — {row['Price [LKR]']}, "
        f"{row['Stock Status']}, {row['RAM']} RAM, {row['Battery Capacity']} battery"
    )


def _phone_not_found(query: str) -> str:
    return (
        f"I couldn't find a phone matching \"{query}\" in our current catalogue. "
        "Could you double-check the model name or try a different spelling?"
    )


# --------------------------------------------------------------- Handlers --
def handle_price_query(nlu: NluResult, resources: Resources) -> str:
    row = da.best_phone_match(nlu.raw_text, resources)
    if row is None:
        return _phone_not_found(nlu.raw_text)
    return (
        f"The {row['Company Name']} {row['Model Name']} is priced at "
        f"{row['Price [LKR]']} ({row['Stock Status']})."
    )


def handle_availability(nlu: NluResult, resources: Resources) -> str:
    row = da.best_phone_match(nlu.raw_text, resources)
    if row is None:
        return _phone_not_found(nlu.raw_text)
    status = str(row["Stock Status"])
    if status.lower() == "in stock":
        return f"Yes, the {row['Company Name']} {row['Model Name']} is currently In Stock."
    return f"The {row['Company Name']} {row['Model Name']} is currently: {status}."


def handle_phone_search(nlu: NluResult, resources: Resources) -> str:
    matches = da.search_phones(nlu.raw_text, resources, top_n=5)
    if matches.empty:
        return _phone_not_found(nlu.raw_text)
    lines = [_format_phone_line(r) for _, r in matches.iterrows()]
    header = "Here's what I found:" if len(lines) > 1 else "Here's what I found:"
    return header + "\n" + "\n".join(f"- {line}" for line in lines)


def handle_comparison(nlu: NluResult, resources: Resources) -> str:
    parts = da.split_comparison_query(nlu.raw_text)
    seen_keys = set()
    rows: List[pd.Series] = []
    for part in parts:
        row = da.best_phone_match(part, resources)
        if row is None:
            continue
        key = (row["Company Name"], row["Model Name"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(row)

    # Fall back: if splitting by "and"/"vs" produced nothing useful, try
    # matching the whole sentence for multiple candidates.
    if len(rows) < 2:
        matches = da.search_phones(nlu.raw_text, resources, top_n=5)
        for _, row in matches.iterrows():
            key = (row["Company Name"], row["Model Name"])
            if key not in seen_keys:
                seen_keys.add(key)
                rows.append(row)
            if len(rows) >= 2:
                break

    if len(rows) < 2:
        return (
            "I could only find one matching phone for that comparison "
            f"({rows[0]['Company Name']} {rows[0]['Model Name']})."
            if rows
            else _phone_not_found(nlu.raw_text)
        )

    a, b = rows[0], rows[1]
    fields = [
        ("Price", "Price [LKR]"),
        ("RAM", "RAM"),
        ("Front Camera", "Front Camera"),
        ("Back Camera", "Back Camera"),
        ("Processor", "Processor"),
        ("Battery", "Battery Capacity"),
        ("Screen Size", "Screen Size"),
        ("Stock", "Stock Status"),
    ]
    lines = [f"Comparing {a['Company Name']} {a['Model Name']} vs {b['Company Name']} {b['Model Name']}:"]
    for label, col in fields:
        lines.append(f"- {label}: {a[col]}  |  {b[col]}")
    return "\n".join(lines)


def handle_accessory_search(nlu: NluResult, resources: Resources) -> str:
    matches = da.search_accessories(nlu.raw_text, resources, nlu.canonical_entities, top_n=5)
    if matches.empty:
        return (
            "I couldn't find a matching accessory in our current stock. "
            "Could you tell me the accessory type (charger, case, earphones...) "
            "and the phone it's for?"
        )
    lines = []
    for _, row in matches.iterrows():
        lines.append(
            f"- {row['accessory_name']} ({row['brand']}) — LKR {row['price_lkr']}, "
            f"compatible with {row['compatible_with']}, stock: {row['stock']}"
        )
    return "Here's what I found:\n" + "\n".join(lines)


def handle_repair(nlu: NluResult, resources: Resources) -> str:
    matches = da.search_repair(nlu.raw_text, resources, nlu.canonical_entities, top_n=5)
    if matches.empty:
        return (
            "I couldn't find that repair service. We handle things like screen "
            "replacement, battery replacement, charging port repair, and more — "
            "could you specify the issue and your phone's brand?"
        )
    lines = []
    for _, row in matches.iterrows():
        lines.append(
            f"- {row['repair_service']} ({row['supported_brand']}) — "
            f"LKR {row['estimated_price_lkr']}, ~{row['estimated_time']}, "
            f"{row['warranty']} warranty"
        )
    return "Here's what I found:\n" + "\n".join(lines)


def handle_faq(nlu: NluResult, resources: Resources) -> str:
    matches = da.search_faq(nlu.raw_text, resources, top_n=1)
    if matches.empty:
        return (
            "I don't have that in our FAQ list yet. Could you rephrase, or ask "
            "about warranty, delivery, store hours, payment, or returns?"
        )
    row = matches.iloc[0]
    return str(row["answer"])


def handle_payment(nlu: NluResult, resources: Resources) -> str:
    methods = nlu.canonical_entities.get("payment_method")
    if not methods:
        lookup = resources.entities_df
        methods = sorted(
            lookup.loc[lookup["entity_type"] == "payment_method", "entity_value"].unique()
        )
    if not methods:
        return "I don't have payment method details available right now."
    return "We accept the following payment methods: " + ", ".join(methods) + "."


def handle_recommendation(nlu: NluResult, resources: Resources) -> str:
    df = resources.phones_df.copy()
    text_low = nlu.raw_text.lower()
    specs = nlu.canonical_entities.get("specification_attribute", [])

    def top_by(col: str, label: str, ascending: bool = False) -> str:
        working = df.copy()
        working["_score"] = working[col].apply(_parse_number)
        working = working.dropna(subset=["_score"]).sort_values("_score", ascending=ascending)
        top = working.head(3)
        lines = [_format_phone_line(r) for _, r in top.iterrows()]
        return f"Based on {label}, here are my top picks:\n" + "\n".join(f"- {l}" for l in lines)

    if "budget" in text_low or "cheap" in text_low or "affordable" in text_low or "price" in nlu.canonical_entities.get("price", []):
        return top_by("Price [LKR]", "price (lowest first)", ascending=True)
    if "Battery" in specs or "battery" in text_low:
        return top_by("Battery Capacity", "battery capacity")
    if "Camera" in specs or "camera" in text_low:
        return top_by("Back Camera", "camera resolution")
    if "RAM" in specs or "ram" in text_low:
        return top_by("RAM", "RAM")

    matches = da.search_phones(nlu.raw_text, resources, top_n=3)
    if not matches.empty:
        lines = [_format_phone_line(r) for _, r in matches.iterrows()]
        return "Here are some options that might fit:\n" + "\n".join(f"- {l}" for l in lines)

    top = df.head(3)
    lines = [_format_phone_line(r) for _, r in top.iterrows()]
    return "Here are some popular options:\n" + "\n".join(f"- {l}" for l in lines)


_HANDLERS = {
    "greeting": lambda nlu, res: GREETING_REPLIES,
    "goodbye": lambda nlu, res: GOODBYE_REPLIES,
    "thanks": lambda nlu, res: THANKS_REPLIES,
    "price_query": handle_price_query,
    "availability": handle_availability,
    "phone_search": handle_phone_search,
    "comparison": handle_comparison,
    "accessory_search": handle_accessory_search,
    "repair": handle_repair,
    "faq": handle_faq,
    "payment": handle_payment,
    "recommendation": handle_recommendation,
}


def generate_response(message: str) -> str:
    message = (message or "").strip()
    if not message:
        return "Please type a question and I'll do my best to help!"

    resources = get_resources()
    nlu = classify(message)
    handler = _HANDLERS.get(nlu.intent.intent)
    if handler is None:
        # Even on "unknown", try FAQ as a last resort - many real
        # questions phrase things the intent model was never trained on.
        faq_matches = da.search_faq(message, resources, top_n=1)
        if not faq_matches.empty:
            return str(faq_matches.iloc[0]["answer"])
        return UNKNOWN_REPLY

    try:
        return handler(nlu, resources)
    except Exception as exc:  # never crash the chat on a data/format issue
        return f"Sorry, something went wrong while looking that up ({exc})."
