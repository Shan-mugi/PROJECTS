# Project 5: Sentiment Analysis on IMDB Reviews (NLP)

## 📌 Project Overview
This project applies Natural Language Processing (NLP) to classify 50,000 movie reviews as either Positive or Negative. It demonstrates the ability to handle unstructured text data.

## ⚙️ NLP Pipeline
1. **Preprocessing:** Removed HTML tags, special characters, and "Stopwords."
2. **Stemming:** Used `PorterStemmer` to reduce words to their root form.
3. **Vectorization:** Applied **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert text into numerical features.
4. **Classification:** Trained a **Multinomial Naive Bayes** classifier.

## 📊 Results
The model achieved a balanced performance across both sentiment classes:
* **Accuracy:** 85%
* **F1-Score:** 0.85

![Sentiment Output](sentiment_output.png)

## 💡 Key Learnings
* Understanding the importance of text cleaning in NLP.
* Converting human language into machine-readable math via TF-IDF.
* Evaluating model performance using precision-recall metrics instead of just raw accuracy.
