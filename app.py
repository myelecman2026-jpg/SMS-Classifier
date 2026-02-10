import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

# Function for text preprocessing
def text_preprocessing(text):
    # lower case
    text = text.lower()
    # tokenization
    text = nltk.word_tokenize(text)

    # removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # removing stop words and punctuation
    stop_words = set(stopwords.words('english'))
    y = []
    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    # stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# load the necessary file
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    # 1. preprocess
    transformed_sms = text_preprocessing(input_sms)
    # 2. vectorize and convert to dense to match model expectations
    vector_input = tfidf.transform([transformed_sms]).toarray()
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
