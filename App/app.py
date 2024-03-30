import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def transform_text(text):
    # TO LOWER CASE:
    text = text.lower()

    # FOT TOKENIZATION:
    text = nltk.word_tokenize(text)

    # REMOVING SPECIAL CHARACTERS:
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # FOR REMOVING STOP WORDS AND PUNCTUATIONS:
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # FOR STEMMING:
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

st.title("Spam Review Detector")

input_review = st.text_area("Enter the Review")

if st.button('Predict'):
    #1. Preprocess:
    transformed_review = transform_text(input_review)

    #2. Vectorize:
    vector_input = tfidf.transform([transformed_review])

    #3. Predict:
    result = model.predict(vector_input)[0]

    #4. Display:
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
