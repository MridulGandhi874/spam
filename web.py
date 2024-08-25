import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


ps = PorterStemmer()

tf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


def preprocessing(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


st.title("Email/SmS Spam Classifier")
input_msg = st.text_input("Enter Your Message: ")

if st.button("Predict"):
    final_msg = preprocessing(input_msg)

    vector_input = tf.transform([final_msg])

    result = model.predict(vector_input)[0]

    if result == 0:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")
