# Disaster-Relief-coordination-system

This project focuses on classifying social media posts during crisis events into relevant and non-relevant tweets, and further into specific informative categories. The aim is to assist emergency responders and crisis managers by automatically filtering and categorizing large volumes of tweets during emergencies.

## 📌 Objective

- Automatically detect whether a tweet is relevant to a crisis.
- Further classify relevant tweets into specific categories (e.g., request for help, infrastructure damage, etc.).
- Explore real-time tweet classification using the Twitter API.

## 🗂️ Dataset

We use the **CrisisNLP** dataset, which includes annotated tweets from six real-world crisis events. Key features:
- Pre-labeled tweets with relevance and specific category tags.
- Diverse linguistic and situational content reflecting real-time crisis data.

## 🛠️ Project Pipeline

- Preprocessed tweet text for traditional ML using lowercasing, punctuation removal, emoji stripping, and stopword removal.
- Minimal preprocessing applied for BERT to retain contextual information.
- Implemented and compared multiple models:
  - **Naive Bayes** and **SVM** with TF-IDF features.
  - **BERT (base-uncased)** with additional dense layers, dropout, and batch normalization.
- Addressed class imbalance using:
  - SMOTE (Synthetic Minority Oversampling Technique)
  - Cost-sensitive learning
- Evaluated models using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

## ⚙️ Technologies Used

- Python 3.x  
- scikit-learn  
- pandas, NumPy  
- Transformers
- matplotlib, seaborn  

## ✅ Key Outcomes

- BERT outperformed traditional models in both binary and multi-class classification tasks.
- Two-stage classification (Relevance → Category) yielded better results than a direct single-step approach.
- Real-time integration introduced challenges in data cleaning, processing latency, and API rate limits.

## 🔄 Future Enhancements

- Replace two-stage classification with a single-stage model for performance and simplicity comparison.
- Improve real-time classification efficiency and interface design.
- Expand support for multilingual and multimodal (image + text) crisis data.
