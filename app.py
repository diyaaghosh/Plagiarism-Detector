import streamlit as st
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize


nltk.download('punkt')

with open("word2vec_model.pkl", "rb") as f:
    word2vec = pickle.load(f)  

with open("best_model.pkl", "rb") as f:
    dnn_model = pickle.load(f)  

def text_to_vec(text, model, embedding_dim=100):
    tokens = word_tokenize(text.lower())
    vecs = []
    for word in tokens:
        if word in model.wv:   
            vecs.append(model.wv[word])
    if len(vecs) == 0:
        return np.zeros(embedding_dim)
    return np.mean(vecs, axis=0)   


st.title("Plagiarism Detector 🔍")
st.write("Check if two texts are plagiarized using a Deep Neural Network with Word2Vec embeddings.")


source_text = st.text_area("Enter Source Text")
suspect_text = st.text_area("Enter Suspected Text")


if st.button("Detect"):
    if source_text.strip() == "" or suspect_text.strip() == "":
        st.warning("⚠️ Please enter both texts.")
    else:
        
        source_vec = text_to_vec(source_text, word2vec)
        suspect_vec = text_to_vec(suspect_text, word2vec)

        # Combine vectors
        features = np.concatenate([source_vec, suspect_vec]).reshape(1, -1)

        # Reshape for model input if necessary
        features = features.reshape((features.shape[0], 1, features.shape[1]))

        # Predict using DNN
        pred = dnn_model.predict(features)

        # If keras model: pred might be probability → round to 0/1
        if pred.ndim > 1:
            pred = (pred > 0.5).astype(int)

        # Show Result
        if pred[0] == 1:
            st.error("🚨 Plagiarism Detected!")
        else:
            st.success("✅ No Plagiarism Detected.")
            st.balloons()
