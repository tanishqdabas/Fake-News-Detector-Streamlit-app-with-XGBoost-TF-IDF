import streamlit as st
import json
import joblib
import os

# Load model and vectorizer using joblib
loaded_model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def vectorisation(input):
    vectorized_text = vectorizer.transform([input])
    return vectorized_text

def load_lottie(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as file:
        return json.load(file)

# Try to load lottie animations (won't crash if files are missing)
lottie_top = load_lottie("2.json")
lottie_side = load_lottie("1.json")
lottie_bottom = load_lottie("4.json")

# Only import streamlit_lottie if we have at least one animation
if any([lottie_top, lottie_side, lottie_bottom]):
    from streamlit_lottie import st_lottie

st.title("Machine Learning is the Future Shield to Fake News")

if lottie_top:
    st_lottie(lottie_top, reverse=True, height=700, width=700, speed=1, loop=True, quality='high', key='top')

st.subheader("Welcome to the Revolution")
st.title("FAKE NEWS DETECTOR")

with st.container():
    st.write("-----")
    left_column, right_column = st.columns(2)

    with left_column:
        st.header("EXPLORE LEGITIMACY OF NEWS")
        st.write("##")
        st.write("Enter news to analyse")
        text = st.text_input("Type your news here")
        st.write("##")
        if st.button("Submit", type="primary"):
            vectorized = vectorisation(text)
            pred = loaded_model.predict(vectorized)
            if pred == 0:
                st.success("✅ The news is classified as: REAL")
            else:
                st.error("🚨 The news is classified as: FAKE")

    with right_column:
        st.write("##")
        if lottie_side:
            st_lottie(lottie_side, reverse=True, height=700, width=700, speed=1, loop=True, quality='high', key='side')

st.title("Machine Learning is the Future Shield to Fake News")

if lottie_bottom:
    st_lottie(lottie_bottom, reverse=True, height=700, width=700, speed=1, loop=True, quality='high', key='bottom')
