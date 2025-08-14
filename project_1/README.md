# Netflix Titles — Category Classifier (Movie vs TV Show)

A clean, portfolio-ready ML project: **predict whether a title is a Movie or a TV Show** using metadata (title/description text, country, rating, duration, etc.).

## 🚀 TL;DR
- Problem: Binary classification (`Category`: Movie vs TV Show)
- Model: Logistic Regression with TF‑IDF (Title/Description) + One‑Hot (categoricals) + numeric features
- Artifacts: Reproducible notebook, saved model, CLI predictor
- Metrics (fill from your run): `Accuracy = ___`, `F1 (TV Show) = ___`, `ROC AUC = ___`

---

## 1) Project structure
```
.
├── data/
│   └── netflex_dataset.csv                  # (not committed) your dataset
├── models/
│   ├── category_clf.joblib          # saved model (created by notebook)
│   └── dir_freq_map.joblib          # director frequency map (created by notebook)
├── notebooks/
│   └── 01_explore_fixed.ipynb       # end‑to‑end training & evaluation
├── src/
│   └── features.py                  # feature builder for training & inference
├── predict.py                       # CLI to run predictions on a JSON/row
├── requirements.txt
├── .gitignore
└── README.md
```
---

## 2) Dataset
- Columns: `Title, Director, Cast, Country, Release_Date, Rating, Duration, Type, Description, Category, ...`
- Target: `Category` ∈ {`Movie`, `TV Show`}

---

## 3) Features engineered
- **Text:** TF‑IDF on `Title`, `Description` (1–2 grams)
- **Dates:** `Year`, `Month` from `Release_Date`
- **Duration:** `Duration_Min` (minutes), `Seasons` (for series)
- **Structure:** `DescLen`, `TitleLen`, `Cast_Count`, `Has_Director`
- **Categoricals:** `Country` (first value), `Rating`, `Type`
- **Frequency:** `Director_Freq` (count of titles per director in training set)

---

## 4) Modeling
- Train/test split (stratified 80/20)
- Pipeline: `ColumnTransformer` + `LogisticRegression` (balanced class weight)
- Evaluation: Accuracy, F1 (TV Show), ROC AUC, confusion matrix
- Model saved to `models/category_clf.joblib` + `dir_freq_map.joblib`

---

## 5) How to run locally

### Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Train & evaluate
Open and run the notebook top to bottom:
```
notebooks/01_explore.ipynb
```
Copy the final **metrics** above into this README.

### Predict from CLI
After running the notebook (which saves the model):

```bash
python predict.py \
  --title "07:19" \
  --description "desperately to stay alive." \
  --country "Mexico" --rating "TV-MA" --type "International Movies" \
  --duration "93 min" --director "Jorge Michel Grau" \
  --cast "Carmen Beato" --release_date "December 23, 2016"
```

Output (example):
```json
{"prediction": "Movie", "probabilities": {"Movie": 0.88, "TV Show": 0.12}}
```

Or pass a JSON file with rows:
```bash
python predict.py --json_file sample_inputs.json
```

---

## 6) Model Card (brief)
- **Intended use:** Educational portfolio project to classify title category from metadata.
- **Training data:** Provided Netflix-like titles dataset.
- **Performance:** Report test Accuracy/F1/ROC AUC above.
- **Limitations:** Metadata can be noisy; region/language biases may exist.
- **Ethical considerations:** Do not use for ranking or content moderation decisions.

---

## 7) Reproducibility & Versioning
- Pin dependencies (see `requirements.txt`)
- Save artifacts with versions, e.g., `models/category_clf__v1.joblib`
- Note dataset version/source in commits

---

## 8) Future work
- Add misclassification analysis & SHAP/coeff inspection
- Try Linear SVM/SGDClassifier, XGBoost
- Small Streamlit demo UI
- Monitor data drift (Evidently) if used in production

---

## 9) License
MIT — feel free to reuse with attribution.
