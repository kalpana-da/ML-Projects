import argparse, json, joblib, pandas as pd
from src.features import build_features

MODEL_PATH = "models/category_clf.joblib"

def predict_df(df: pd.DataFrame):
    model = joblib.load(MODEL_PATH)
    X = build_features(df)
    proba = model.predict_proba(X)
    preds = model.predict(X)
    classes = list(model.classes_)
    results = []
    for i in range(len(df)):
        p = {classes[j]: float(proba[i, j]) for j in range(len(classes))}
        results.append({"prediction": preds[i], "probabilities": p})
    return results

def main():
    parser = argparse.ArgumentParser(description="Predict Movie vs TV Show from metadata.")
    parser.add_argument("--json_file", type=str, help="Path to a JSON file containing a list of rows.")
    parser.add_argument("--title")
    parser.add_argument("--description")
    parser.add_argument("--country")
    parser.add_argument("--rating")
    parser.add_argument("--type")
    parser.add_argument("--duration")
    parser.add_argument("--director")
    parser.add_argument("--cast")
    parser.add_argument("--release_date")
    args = parser.parse_args()

    if args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        row = {
            "Title": args.title,
            "Description": args.description,
            "Country": args.country,
            "Rating": args.rating,
            "Type": args.type,
            "Duration": args.duration,
            "Director": args.director,
            "Cast": args.cast,
            "Release_Date": args.release_date,
        }
        df = pd.DataFrame([row])

    results = predict_df(df)
    print(json.dumps(results if args.json_file else results[0], ensure_ascii=False))

if __name__ == "__main__":
    main()
