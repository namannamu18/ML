import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix

df = pd.read_csv('sample_text_data.csv')
print(df.head())

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2, random_state=42)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print('Accuracy: ',accuracy_score(y_test, y_pred))
print("CM",confusion_matrix(y_test,y_pred))