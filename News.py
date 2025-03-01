import requests
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"

def get_stock_symbol(company_name):
    search_url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={API_KEY}"
    response = requests.get(search_url).json()    
    best_match = response.get("bestMatches", [])
    return best_match[0]["1. symbol"] 
  

company_name = input("Enter company name: ").strip()

company_symbol = get_stock_symbol(company_name)


news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={company_symbol}&apikey={API_KEY}"
response = requests.get(news_url).json()
x = []
for i, article in enumerate(response.get("feed", [])):
    if i >= 5: break
    x.append(article['title'])
ps = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    words = text.lower().split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

x = [preprocess_text(doc) for doc in x]

# Load models
vectorizer = joblib.load('tfidf_vectorizer.pkl')
encoder = joblib.load('label_encoder.pkl')
model = joblib.load('LogisticRegression()_model.pkl')

# Transform input using pre-trained vectorizer
x = vectorizer.transform(x).toarray()

# Predict sentiment
y_pred = model.predict(x)
y_pred = encoder.inverse_transform(y_pred)

print('Predicted sentiment:', y_pred)
