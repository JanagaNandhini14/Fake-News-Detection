# 📰 Fake News Detection

This project uses **Logistic Regression** with **TF-IDF Vectorization** to classify news articles as **Fake** or **True**.  
The dataset is based on the Kaggle Fake News dataset.

## 🚀 Features
- Preprocesses text data using **TF-IDF**
- Splits dataset into training and testing sets
- Trains a Logistic Regression model
- Evaluates with accuracy score

## 📂 Dataset
- `Fake.csv` → Fake news articles
- `True.csv` → Real news articles

## ⚡ Results
Achieved an accuracy of ~90% on the test set.

## 🔧 Requirements
Install dependencies:
```bash
pip install pandas scikit-learn
