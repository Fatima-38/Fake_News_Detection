# 📰 Fake News Detection using NLP & Machine Learning

Detects whether a news headline/article is **REAL** or **FAKE** using TF-IDF
text vectorization and classic machine learning classifiers.

## 🧠 Overview

| | |
|---|---|
| **Domain** | Natural Language Processing, Machine Learning |
| **Models used** | Logistic Regression, Multinomial Naive Bayes |
| **Technique** | TF-IDF vectorization + text cleaning |
| **Dataset** | Sample (240 rows) included — see [Dataset](#-dataset) for real data |

## 📁 Project Structure
```
Fake_News_Detection/
├── fake_news_detector.py     # main script
├── dataset.csv                # sample data
├── requirements.txt
├── outputs/                   # generated charts (after running)
└── README.md
```

## ⚙️ Setup & Run
```bash
pip install -r requirements.txt
python fake_news_detector.py
```

## 📊 What it does
1. Loads and cleans news text (removes URLs, punctuation, numbers, stopwords)
2. Converts text to numeric features using **TF-IDF**
3. Trains **Logistic Regression** and **Naive Bayes** classifiers
4. Evaluates with accuracy, precision/recall, and confusion matrices
5. Predicts on new/unseen headlines

## 📈 Sample Output
```
=== Logistic Regression ===
Accuracy: 100.0%
              precision    recall  f1-score
        FAKE       1.00      1.00      1.00
        REAL       1.00      1.00      1.00
```
Confusion matrices are saved to `outputs/`.

## 🗂 Dataset
`dataset.csv` bundled here is a **small synthetic sample** (240 templated
rows) so the project runs instantly with no setup, which is why accuracy
looks artificially perfect. For real-world results, swap it with the
**Fake and Real News Dataset** (44,000+ real articles):

🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Just keep the same columns: `title`, `text`, `label` (values `REAL`/`FAKE`).

## 🚀 Possible Extensions
- Swap TF-IDF for word embeddings (Word2Vec / GloVe / BERT)
- Deploy as a Streamlit web app for live headline checking
- Add sentiment analysis as an additional feature
