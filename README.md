# Mobixa Chatbot

A mobile-shop chatbot built on your existing trained models and datasets.

## Install

```bash
cd "Mobixa Chat bot"
pip install -r requirements.txt
```

(NLTK corpora — punkt, stopwords, wordnet — are downloaded automatically on first run.)

## Run

```bash
streamlit run src/app.py
```

## How it works

`User message → preprocessing (src/preprocessing.py) → intent classification
(src/nlu.py, DL model primary / ML model fallback + rule overrides) → entity
extraction (entity_dictionary.pkl) → dataset lookup (src/data_access.py) →
response (src/responder.py)`

- **Intent classification**: `intent_classifier_dl.keras` (LSTM + tokenizer +
  label_encoder) is the primary classifier; `intent_classifier_ml.pkl` (SVM +
  `tfidf_vectorizer.pkl`) is used as a fallback when DL confidence is low.
  Repair and accessory queries — not present in the original intent training
  data — are detected via `entity_dictionary.pkl` matches.
- **Entities**: reuses `entity_dictionary.pkl` exactly as built in
  `06_NER_Model.ipynb`, plus a canonical lookup built from `entities.csv` to
  map keywords (e.g. "charger") to dataset labels (e.g. "Charger").
- **Responses**: always generated from the real CSVs
  (`Phones_Dataset_clean.csv`, `mobile_accessories_clean.csv`,
  `repair_services_clean.csv`, `mobixa_faq_clean.csv`) — nothing is invented.

## Known limitations

- The intent classifiers were trained on a small dataset (10 intents, ~100
  examples each) and can occasionally misclassify unusual phrasing; the app
  never crashes on this, it degrades to "not found" / "unknown" responses.
- The phone dataset includes tablets under the same catalogue, so
  "recommend a phone" style queries can occasionally surface a tablet.
