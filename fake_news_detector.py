import re
import string
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay


# ---------- 1. Load data ----------
def load_data(path="dataset.csv"):
    df = pd.read_csv(path)
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    return df


# ---------- 2. Clean text ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)                       # remove numbers
    text = re.sub(r"\s+", " ", text).strip()               # collapse whitespace
    return text


# ---------- 3. Train + evaluate ----------
def train_and_evaluate(df):
    df["clean_content"] = df["content"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
    }

    results = {}
    best_model_name, best_model, best_acc = None, None, -1

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.2%}")
        print(classification_report(y_test, preds))

        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model

        # Confusion matrix plot
        cm = confusion_matrix(y_test, preds, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        fname = f"outputs/confusion_matrix_{name.replace(' ', '_').lower()}.png"
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"Saved: {fname}")

    print(f"\nBest model: {best_model_name} ({best_acc:.2%} accuracy)")
    return vectorizer, best_model, best_model_name, results


# ---------- 4. Predict new headlines ----------
def predict_news(vectorizer, model, texts):
    cleaned = [clean_text(t) for t in texts]
    vec = vectorizer.transform(cleaned)
    preds = model.predict(vec)
    probs = model.predict_proba(vec) if hasattr(model, "predict_proba") else None
    for i, t in enumerate(texts):
        label = preds[i]
        conf = f" (confidence: {max(probs[i]):.1%})" if probs is not None else ""
        print(f"\n Headline: {t}\n Prediction: {label}{conf}")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("Loading dataset...")
    df = load_data("dataset.csv")
    print(f"Loaded {len(df)} rows -> {df['label'].value_counts().to_dict()}")

    vectorizer, model, model_name, results = train_and_evaluate(df)

    print("\n\n===== Testing on new/unseen headlines =====")
    sample_headlines = [
        "Government announces new education policy for schools nationwide",
        "SHOCKING: Celebrity secretly reveals miracle cure doctors HATE, share before deleted",
        "Local council approves budget for road infrastructure repairs",
        "BREAKING: Whistleblower says aliens control the stock market, government denies",
    ]
    predict_news(vectorizer, model, sample_headlines)
