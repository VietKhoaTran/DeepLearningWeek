import pandas as pd
import joblib

df = pd.read_csv('Data.csv', encoding='ISO-8859-1')
x = df.iloc[:, 1]
y = df.iloc[:, 0]

# Preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    words = text.lower().split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    words = ' '.join(words)
    return words
x = [preprocess_text(doc) for doc in x]

# TFIDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features = 5000)
x = vectorizer.fit_transform(x).toarray()
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Labelencoding y
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)
joblib.dump(encoder, "label_encoder.pkl")

# Try different models and keep the one with highest accuracy
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(5000,)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Accuracy for Neural Network model: {accuracy: .2f}")
model.save('NN_moidel.h5')

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

list = [LogisticRegression(), MultinomialNB(), SVC()]
for i in list:
    i.fit(x_train, y_train)
    y_pred = i.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {i}: {accuracy: .2f}")
    joblib.dump(i, f'{i}_model.pkl')
# Since the logistic has the highest accuracy, we will use it for later taks