"""
All lookups against the real Mobixa datasets. Every function here reads
straight from the CSVs loaded once in model_loader.Resources - no
hard-coded or invented data is ever returned.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

import pandas as pd

from .model_loader import Resources

_SIMPLE_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "do", "does", "did", "for", "of",
    "in", "on", "at", "to", "how", "much", "what", "whats", "your", "you",
    "have", "has", "i", "want", "need", "please", "can", "could", "would",
    "me", "my", "it", "its", "with", "and", "or", "vs", "versus", "compare",
    "comparison", "between", "cost", "costs", "price", "about", "tell",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _content_tokens(text: str) -> set:
    return {t for t in _tokenize(text) if t not in _SIMPLE_STOPWORDS}


def _row_tokens(row: pd.Series, columns: Iterable[str]) -> set:
    parts = []
    for c in columns:
        val = row.get(c)
        if pd.notna(val):
            parts.append(str(val))
    return set(_tokenize(" ".join(parts)))


def _rank_rows(
    df: pd.DataFrame, columns: Iterable[str], query_tokens: set, top_n: int = 5
) -> pd.DataFrame:
    if not query_tokens or df.empty:
        return df.iloc[0:0]

    scored = []
    for idx, row in df.iterrows():
        row_tok = _row_tokens(row, columns)
        if not row_tok:
            continue
        overlap = query_tokens & row_tok
        if not overlap:
            continue
        coverage = len(overlap) / len(row_tok)
        score = len(overlap) + coverage
        scored.append((score, len(row_tok), idx))

    if not scored:
        return df.iloc[0:0]

    scored.sort(key=lambda x: (-x[0], x[1]))
    top_idx = [s[2] for s in scored[:top_n]]
    return df.loc[top_idx]


# ---------------------------------------------------------------- Phones --
def search_phones(query: str, resources: Resources, top_n: int = 5) -> pd.DataFrame:
    tokens = _content_tokens(query)
    return _rank_rows(
        resources.phones_df, ["Company Name", "Model Name"], tokens, top_n
    )


def best_phone_match(query: str, resources: Resources) -> Optional[pd.Series]:
    matches = search_phones(query, resources, top_n=1)
    if matches.empty:
        return None
    return matches.iloc[0]


def split_comparison_query(text: str) -> List[str]:
    parts = re.split(r"\bvs\.?\b|\bversus\b|\band\b|,", text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


# ----------------------------------------------------------- Accessories --
def search_accessories(
    query: str,
    resources: Resources,
    canonical_entities: Optional[dict] = None,
    top_n: int = 5,
) -> pd.DataFrame:
    df = resources.accessories_df
    canonical_entities = canonical_entities or {}

    # Narrow down using canonical entity matches first (precise).
    filtered = df
    accessory_types = canonical_entities.get("accessory_type")
    if accessory_types:
        filtered = filtered[filtered["category"].isin(accessory_types)]

    brands = canonical_entities.get("brand")
    if brands and not filtered.empty:
        brand_mask = filtered["brand"].isin(brands) | filtered["compatible_with"].apply(
            lambda v: any(b.lower() in str(v).lower() for b in brands)
        )
        brand_filtered = filtered[brand_mask]
        if not brand_filtered.empty:
            filtered = brand_filtered

    if not filtered.empty:
        return filtered.head(top_n)

    # Fall back to free-text token overlap search across all fields.
    tokens = _content_tokens(query)
    return _rank_rows(
        df,
        ["accessory_name", "category", "brand", "compatible_with"],
        tokens,
        top_n,
    )


# ----------------------------------------------------------------- Repair --
def search_repair(
    query: str,
    resources: Resources,
    canonical_entities: Optional[dict] = None,
    top_n: int = 5,
) -> pd.DataFrame:
    df = resources.repair_df
    canonical_entities = canonical_entities or {}

    filtered = df
    repair_types = canonical_entities.get("repair_type")
    if repair_types:
        filtered = filtered[filtered["repair_service"].isin(repair_types)]

    brands = canonical_entities.get("brand")
    if brands and not filtered.empty:
        brand_filtered = filtered[filtered["supported_brand"].isin(list(brands) + ["Universal", "Android"])]
        if not brand_filtered.empty:
            filtered = brand_filtered

    if not filtered.empty:
        return filtered.head(top_n)

    tokens = _content_tokens(query)
    return _rank_rows(
        df, ["repair_service", "supported_brand", "description"], tokens, top_n
    )


# -------------------------------------------------------------------- FAQ --
def search_faq(query: str, resources: Resources, top_n: int = 3) -> pd.DataFrame:
    tokens = _content_tokens(query)
    return _rank_rows(resources.faq_df, ["question", "category"], tokens, top_n)
