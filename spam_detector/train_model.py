import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

data = {
    'text': [
        "Congratulations! You won a $1,000 Walmart gift card. Click here to claim now!",
        "Hey, are we still meeting for lunch today at 1 PM?",
        "URGENT: Your account has been compromised. Verify your password immediately.",
        "Hi Mom, I will be home late tonight. Save some dinner for me.",
        "Win FREE iPhone 15 now! Limited time offer, click link below.",
        "Please find attached the report for yesterday's meeting.",
        "Get cheap loans with 0% interest rate guaranteed!",
        "Can you send me the updated project guidelines when you're free?"
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['text']
y = df['label_num']

vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

print("Model trained successfully!")

joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model files (spam_model.pkl & vectorizer.pkl) saved!")
