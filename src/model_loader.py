"""
Loads every trained model / vectorizer / dataset exactly once and
exposes them through a single cached `Resources` object.

No absolute paths are used - everything is resolved relative to the
project root (the parent of this `src/` folder), so the project can be
moved or deployed anywhere.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
MODELS_DIR = PROJECT_ROOT / "models"


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@dataclass
class Resources:
    # ML intent model
    tfidf_vectorizer: object
    bow_vectorizer: object
    intent_classifier_ml: object

    # DL intent model
    tokenizer: object
    label_encoder: object
    intent_classifier_dl: object
    dl_max_len: int

    # NER
    entity_dictionary: dict
    # (entity_type, synonym_or_value_lowercase) -> canonical entity_value
    entity_lookup: dict

    # Datasets
    phones_df: pd.DataFrame
    accessories_df: pd.DataFrame
    faq_df: pd.DataFrame
    intents_df: pd.DataFrame
    repair_df: pd.DataFrame
    entities_df: pd.DataFrame


@lru_cache(maxsize=1)
def get_resources() -> "Resources":
    """Load every artifact once per process and cache the result."""

    # --- ML model artifacts ---
    tfidf_vectorizer = _load_pickle(MODELS_DIR / "tfidf_vectorizer.pkl")
    bow_vectorizer = _load_pickle(MODELS_DIR / "bow_vectorizer.pkl")
    intent_classifier_ml = _load_pickle(MODELS_DIR / "intent_classifier_ml.pkl")

    # --- DL model artifacts ---
    tokenizer = _load_pickle(MODELS_DIR / "tokenizer.pkl")
    label_encoder = _load_pickle(MODELS_DIR / "label_encoder.pkl")

    from tensorflow import keras

    intent_classifier_dl = keras.models.load_model(
        MODELS_DIR / "intent_classifier_dl.keras"
    )
    # The model was trained with a fixed input length equal to the
    # longest padded sequence at training time. Read it straight from
    # the model's own input shape so we never hard-code a number that
    # could drift out of sync with the saved model.
    dl_max_len = int(intent_classifier_dl.input_shape[1])

    # --- NER ---
    entity_dictionary = _load_pickle(MODELS_DIR / "entity_dictionary.pkl")

    # --- Datasets ---
    phones_df = _load_csv(DATA_DIR / "Phones_Dataset_clean.csv")
    accessories_df = _load_csv(DATA_DIR / "mobile_accessories_clean.csv")
    faq_df = _load_csv(DATA_DIR / "mobixa_faq_clean.csv")
    intents_df = _load_csv(DATA_DIR / "mobixa_intents_clean.csv")
    repair_df = _load_csv(DATA_DIR / "repair_services_clean.csv")
    entities_df = _load_csv(DATA_DIR / "entities.csv")

    # Build a (entity_type, synonym/value) -> canonical entity_value lookup
    # so extracted keywords (e.g. "charger") can be mapped back to the
    # canonical dataset label (e.g. "Charger") for precise filtering.
    entity_lookup: dict = {}
    for _, row in entities_df.iterrows():
        etype = row.get("entity_type")
        evalue = row.get("entity_value")
        if pd.isna(etype) or pd.isna(evalue):
            continue
        candidates = [str(evalue).strip().lower()]
        syn = row.get("synonyms")
        if pd.notna(syn):
            candidates.extend(s.strip().lower() for s in str(syn).split(","))
        for c in candidates:
            if c:
                entity_lookup[(etype, c)] = evalue

    return Resources(
        tfidf_vectorizer=tfidf_vectorizer,
        bow_vectorizer=bow_vectorizer,
        intent_classifier_ml=intent_classifier_ml,
        tokenizer=tokenizer,
        label_encoder=label_encoder,
        intent_classifier_dl=intent_classifier_dl,
        dl_max_len=dl_max_len,
        entity_dictionary=entity_dictionary,
        entity_lookup=entity_lookup,
        phones_df=phones_df,
        accessories_df=accessories_df,
        faq_df=faq_df,
        intents_df=intents_df,
        repair_df=repair_df,
        entities_df=entities_df,
    )
