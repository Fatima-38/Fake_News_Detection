# 📰 Fake News Detection using NLP & Machine Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)](https://scikit-learn.org/)
[![NLP](https://img.shields.io/badge/Domain-Natural_Language_Processing-purple.svg)]()

A robust Natural Language Processing (NLP) and Machine Learning system that classifies news articles and headlines as **REAL** or **FAKE** using TF-IDF feature extraction, text preprocessing, and benchmarked statistical classifiers.

---

## 📌 Overview

| Attribute | Specification |
|---|---|
| **Domain** | Natural Language Processing (NLP), Text Classification, Disinformation Detection |
| **Algorithms** | Logistic Regression (L2 Regularization), Multinomial Naive Bayes |
| **Feature Extraction** | TF-IDF (Term Frequency - Inverse Document Frequency) N-Gram Vectorizer |
| **Evaluation Metrics** | Accuracy, Precision, Recall, Macro F1-Score, Confusion Matrix |

---

## ⚙️ NLP Pipeline Workflow

1. **Text Preprocessing:** Lowercasing, URL removal, punctuation stripping, number normalization, and NLTK stopword filtering.
2. **Feature Vectorization:** Converting text corpora into high-dimensional TF-IDF sparse matrices.
3. **Model Training:** Benchmarking Logistic Regression against Multinomial Naive Bayes under stratified train-test splits.
4. **Model Evaluation:** Computing classification reports and exporting visual confusion matrices to `outputs/`.
5. **Inference Engine:** Interactive testing function predicting on novel, unseen headlines.

---

## 📂 Project Structure

```
Fake_News_Detection/
├── fake_news_detector.py     # Main NLP pipeline & classification script
├── dataset.csv               # Dataset corpus (Title, Text, Label)
├── requirements.txt          # Python package requirements
├── outputs/                  # Exported confusion matrices & evaluation charts
├── LICENSE                   # MIT License
└── README.md                 # Documentation
```

---

## 🚀 Setup & Run

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Fatima-38/Fake_News_Detection.git
cd Fake_News_Detection
pip install -r requirements.txt
```

### 2. Execute Training & Evaluation Pipeline
```bash
python fake_news_detector.py
```

---

## 📊 Sample Output & Performance

```text
=== Classification Report (Logistic Regression) ===
              precision    recall  f1-score   support

        FAKE       0.98      0.97      0.97       120
        REAL       0.97      0.98      0.98       120

    accuracy                           0.98       240
   macro avg       0.98      0.98      0.98       240
weighted avg       0.98      0.98      0.98       240
```
*Visual confusion matrices are automatically rendered and saved to `outputs/`.*

---

## 🛠️ Technologies Used

- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn (LogisticRegression, MultinomialNB, TfidfVectorizer, train_test_split)
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Fatima Javaid**  
- **GitHub:** [@Fatima-38](https://github.com/Fatima-38)  
- **Email:** fatimajavaid503@gmail.com  
- **Portfolio:** [fatima-portfolio](https://fatima-38.github.io/fatima-portfolio/)
