import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import re

# 1. Setup NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# 2. Load Dataset (IMDB)
df = pd.read_csv(r'C:\Users\ELCOT\OneDrive\Desktop\Project_5\IMDB Dataset.csv.zip')

def clean_text(text):
    # Remove HTML tags and special characters
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    # Remove stopwords and apply stemming
    text = [ps.stem(word) for word in text if not word in stop_words]
    return ' '.join(text)

print("Cleaning text data... this may take a minute...")
df['review'] = df['review'].apply(clean_text)

# 3. Vectorization (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['review']).toarray()
y = df['sentiment'].map({'positive': 1, 'negative': 0})

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model: Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("\n--- Sentiment Analysis Report ---")
print(classification_report(y_test, y_pred))
