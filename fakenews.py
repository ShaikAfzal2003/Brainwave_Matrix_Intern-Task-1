import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("fake_real_news_dataset_final_all_domains.csv")
X = df['text']
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model_lr = LogisticRegression()
model_nb = MultinomialNB()
model_pa = PassiveAggressiveClassifier()

model_lr.fit(X_train_tfidf, y_train)
model_nb.fit(X_train_tfidf, y_train)
model_pa.fit(X_train_tfidf, y_train)


print(" Logistic Regression Accuracy:", accuracy_score(y_test, model_lr.predict(X_test_tfidf)))
print(" Naive Bayes Accuracy:", accuracy_score(y_test, model_nb.predict(X_test_tfidf)))
print("Passive Aggressive Accuracy:", accuracy_score(y_test, model_pa.predict(X_test_tfidf)))


def output_label(label):
    return "Real News" if label == "REAL" else "❌ Fake News"


def manual_testing(news):
    vect = vectorizer.transform([news])
    pred_lr = model_lr.predict(vect)[0]
    pred_nb = model_nb.predict(vect)[0]
    pred_pa = model_pa.predict(vect)[0]

    print("\nLogistic Regression Prediction:", output_label(pred_lr))
    print("Naive Bayes Prediction:", output_label(pred_nb))
    print(" Passive Aggressive Prediction:", output_label(pred_pa))


news_input = input("\nEnter the news to verify:\n")
manual_testing(news_input)
