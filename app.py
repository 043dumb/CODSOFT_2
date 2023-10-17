import pickle
import streamlit as st
import nltk
import re
import string

nltk.download('stopwords')

st.title('Movie Genre Predictor')


def clean_text(text):
    text = text.lower()  # Lowercase all characters
    text = re.sub(r'@\S+', '', text)  # Remove Twitter handles
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z+']", ' ', text)  # Keep only characters
    text = "".join([i for i in text if i not in string.punctuation])

    words = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')  # Remove stopwords
    text = " ".join([i for i in words if i not in stopwords and len(i) > 2])
    text = re.sub("\s[\s]+", " ", text).strip()  # Remove repeated/leading/trailing spaces
    return text

temp_name = st.text_input(label='Movie name')

temp_des = st.text_area(label='Movie description')

Name = ""
des = ""
if st.button(label='submit'):
    Name = temp_name
    des = temp_des

    des = clean_text(des)

    tfidf = pickle.load(open('tfidf_movie_genre.pkl', 'rb'))
    model = pickle.load(open('model_movie_genre.pkl', 'rb'))
    le = pickle.load(open('enc.pkl', 'rb'))

    vector_des = tfidf.transform([des])
    res = model.predict(vector_des)
    res = le.inverse_transform(res)
    st.text(body=res[0])
