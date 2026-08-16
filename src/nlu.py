"""
Natural-language understanding: intent classification (using the
existing trained ML + DL models) and rule/dictionary-based entity
extraction (reusing entity_dictionary.pkl exactly as built in
notebooks/06_NER_Model.ipynb).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List

import numpy as np

from .model_loader import Resources, get_resources
from .preprocessing import preprocess_text

# Below this DL softmax confidence, the prediction is treated as unreliable.
DL_CONFIDENCE_THRESHOLD = 0.35

# Intents the trained classifiers actually know about. Repair and FAQ
# queries were not part of the intent training data, so they are
# detected with dictionary/keyword rules layered on top (see
# routing in responder.py).
KNOWN_INTENTS = [
    "greeting",
    "goodbye",
    "thanks",
    "price_query",
    "phone_search",
    "availability",
    "comparison",
    "accessory_search",
    "recommendation",
    "payment",
]

REPAIR_KEYWORDS = [
    "repair", "fix", "broken", "crack", "cracked", "replace", "replacement",
    "not charging", "won't turn on", "wont turn on", "damage", "damaged",
    "screen issue", "battery drain", "swelling", "water damage", "wet phone",
]

FAQ_KEYWORDS = [
    "warranty", "delivery", "deliver", "opening hour", "open hour",
    "return policy", "refund", "exchange", "store location", "branch",
    "shipping", "policy", "hours", "cash on delivery",
]


@dataclass
class IntentResult:
    intent: str
    confidence: float
    source: str  # "dl", "ml", or "rule"


@dataclass
class NluResult:
    raw_text: str
    processed_text: str
    intent: IntentResult
    entities: Dict[str, List[str]] = field(default_factory=dict)
    canonical_entities: Dict[str, List[str]] = field(default_factory=dict)


def canonicalize_entities(
    entities: Dict[str, List[str]], resources: Resources | None = None
) -> Dict[str, List[str]]:
    """Map raw matched keywords (e.g. "charger", "iphone") to their
    canonical dataset labels (e.g. "Charger", "Apple") using the
    entity_type/value/synonym table in entities.csv."""
    resources = resources or get_resources()
    canonical: Dict[str, List[str]] = {}
    for etype, words in entities.items():
        values = set()
        for w in words:
            canon = resources.entity_lookup.get((etype, w))
            if canon:
                values.add(canon)
        if values:
            canonical[etype] = sorted(values)
    return canonical


def extract_entities(text: str, resources: Resources | None = None) -> Dict[str, List[str]]:
    """Exact reproduction of extract_entities() from
    notebooks/06_NER_Model.ipynb, using the pre-built entity_dictionary."""
    resources = resources or get_resources()
    text_low = text.lower()
    detected: Dict[str, List[str]] = {}

    for entity_type, words in resources.entity_dictionary.items():
        found = []
        for word in words:
            if not word:
                continue
            try:
                if re.search(r"\b" + re.escape(word) + r"\b", text_low):
                    found.append(word)
            except re.error:
                continue
        if found:
            detected[entity_type] = found

    return detected


def _predict_intent_dl(processed_text: str, resources: Resources) -> IntentResult:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    seq = resources.tokenizer.texts_to_sequences([processed_text])
    seq = pad_sequences(seq, maxlen=resources.dl_max_len, padding="post")
    probs = resources.intent_classifier_dl.predict(seq, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = resources.label_encoder.inverse_transform([idx])[0]
    return IntentResult(intent=str(label), confidence=float(probs[idx]), source="dl")


def _predict_intent_ml(processed_text: str, resources: Resources) -> IntentResult:
    vector = resources.tfidf_vectorizer.transform([processed_text])
    try:
        label = resources.intent_classifier_ml.predict(vector)[0]
    except Exception:
        # Guards against sklearn version/pickle incompatibilities in the
        # underlying libsvm binding; fall back to "no confident ML vote"
        # rather than crashing the chat.
        return IntentResult(intent="unknown", confidence=0.0, source="ml")
    # SVC was trained without probability=True, so we fall back to the
    # decision function margin as a rough confidence proxy.
    confidence = 1.0
    try:
        margins = resources.intent_classifier_ml.decision_function(vector)[0]
        margins = np.atleast_1d(margins)
        confidence = float(1 / (1 + np.exp(-np.max(margins))))
    except Exception:
        pass
    return IntentResult(intent=str(label), confidence=confidence, source="ml")


def _rule_based_intent(text_low: str, entities: Dict[str, List[str]]) -> IntentResult | None:
    if "repair_type" in entities or any(kw in text_low for kw in REPAIR_KEYWORDS):
        return IntentResult(intent="repair", confidence=1.0, source="rule")
    if any(kw in text_low for kw in FAQ_KEYWORDS):
        return IntentResult(intent="faq", confidence=0.9, source="rule")
    # "accessory_type" is an unambiguous, dictionary-matched signal
    # (charger, case, earphones, ...) that the intent classifiers
    # frequently confuse with phone_search/price_query.
    if "accessory_type" in entities:
        return IntentResult(intent="accessory_search", confidence=0.9, source="rule")
    return None


_SMALL_TALK_INTENTS = {"greeting", "goodbye", "thanks"}


@lru_cache(maxsize=1)
def _small_talk_keywords() -> Dict[str, set]:
    """Data-driven keyword sets for greeting/goodbye/thanks, built from the
    real training texts in mobixa_intents_clean.csv. Used as a sanity check
    so unrelated/out-of-domain sentences don't get misclassified as
    small talk just because the tiny classifier is uncertain."""
    resources = get_resources()
    keywords: Dict[str, set] = {}
    for intent in _SMALL_TALK_INTENTS:
        subset = resources.intents_df[resources.intents_df["intent"] == intent]
        tokens = set()
        for text in subset["processed_text"].dropna():
            tokens.update(str(text).split())
        keywords[intent] = tokens
    return keywords


def classify(text: str) -> NluResult:
    resources = get_resources()
    processed = preprocess_text(text)
    text_low = text.lower()
    entities = extract_entities(text, resources)

    canonical_entities = canonicalize_entities(entities, resources)

    # 1. Rule overrides for intents the classifiers were never trained on
    #    (or frequently confuse), using unambiguous dictionary signals.
    rule_result = _rule_based_intent(text_low, entities)
    if rule_result is not None:
        return NluResult(text, processed, rule_result, entities, canonical_entities)

    processed_tokens = set(processed.split())

    def _passes_small_talk_check(result: IntentResult) -> bool:
        if result.intent not in _SMALL_TALK_INTENTS:
            return True
        return bool(processed_tokens & _small_talk_keywords()[result.intent])

    # 2. Primary: deep-learning classifier (gives a real softmax confidence).
    dl_result = _predict_intent_dl(processed, resources)
    if dl_result.confidence >= DL_CONFIDENCE_THRESHOLD and _passes_small_talk_check(dl_result):
        return NluResult(text, processed, dl_result, entities, canonical_entities)

    # 3. Fallback: ML classifier.
    ml_result = _predict_intent_ml(processed, resources)
    if ml_result.confidence >= DL_CONFIDENCE_THRESHOLD and _passes_small_talk_check(ml_result):
        return NluResult(text, processed, ml_result, entities, canonical_entities)

    # 4. Neither model gives a confident, sane prediction -> unknown.
    best = dl_result if dl_result.confidence >= ml_result.confidence else ml_result
    unknown = IntentResult(intent="unknown", confidence=best.confidence, source=best.source)
    return NluResult(text, processed, unknown, entities, canonical_entities)
