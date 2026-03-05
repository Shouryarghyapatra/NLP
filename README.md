# NLP Text Classification – Flagged vs Not Flagged Detection

## 📌 Project Overview
This project performs text classification using Natural Language Processing (NLP) techniques to classify textual responses into two categories:

- **Flagged (1)**
- **Not Flagged (0)**

The dataset used for this project is sourced from Kaggle and contains text responses along with a binary classification label.

The objective of this project is to preprocess raw text data, convert it into numerical features, and build a machine learning model to accurately classify text entries.

---

## 📊 Dataset Information

- **Source:** Kaggle  
- The dataset contains:
  - `response_text` → The raw text input
  - `class` → Label (`flagged` / `not_flagged`)

### Data Characteristics:
- Binary classification problem
- Text-based dataset
- No missing values detected
- Class distribution visualized using a pie chart

---

## 🎯 Problem Statement

Text moderation and automated flag detection are important in many applications such as:
- Social media platforms
- Content moderation systems
- Online review systems
- Public discussion forums

The goal is to build a machine learning model that can automatically detect whether a text response should be flagged or not.

---

## 🛠 Tools & Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- NLTK
- Scikit-learn

---

## 🔍 Project Workflow

### 1️⃣ Data Loading
- Imported dataset using Pandas
- Checked shape, data types, and null values
- Verified dataset integrity

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Checked class distribution
- Visualized flagged vs not_flagged distribution using pie chart
- Verified dataset balance

---

### 3️⃣ Text Preprocessing

The following preprocessing steps were applied:

✔ Convert text to lowercase  
✔ Remove non-alphabetic characters using Regular Expressions  
✔ Tokenization  
✔ Stopword removal using `ENGLISH_STOP_WORDS`  
✔ Custom simple lemmatization function  

Example preprocessing steps:
- Remove punctuation
- Remove stopwords
- Reduce words like:
  - "studies" → "study"
  - "running" → "run"
  - "children" (basic rule-based reduction)

---

### 4️⃣ Feature Extraction

Used **CountVectorizer** to convert text into numerical features:

- `max_features = 100`
- Selected top 100 most frequent words
- Created Bag-of-Words representation
- Converted sparse matrix to array format

This transforms text into machine-readable numeric vectors.

---

### 5️⃣ Model Building

Used:

**Gaussian Naive Bayes (GaussianNB)**

Steps:
- Split dataset into:
  - 75% training
  - 25% testing
- Trained model using training data
- Predicted results on test data

---

### 6️⃣ Model Evaluation

Evaluation metric used:

- **Accuracy Score** 0.65

```python
metrics.accuracy_score(y_pred, y_test)
