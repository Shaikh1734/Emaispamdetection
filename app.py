
import streamlit as st
import joblib

# Load vectorizer and model
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model_nb.pkl')

st.title("Email Spam Detector")
st.write("Paste your email text below:")

user_input = st.text_area("Email Text:", height=150)

if st.button("Predict"):
    if user_input:
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0].max()
        st.success(f"Prediction: {prediction.upper()}")
        st.info(f"Confidence: {prob:.2f}")
    else:
        st.warning("Please enter some text.")
