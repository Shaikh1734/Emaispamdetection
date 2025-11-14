
import streamlit as st
import joblib

vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('spam_classifier_model.joblib')

label_map = {0: "Ham", 1: "Spam"}

st.title("Email Spam Detector")
st.write("Paste your email text below:")

user_input = st.text_area("Email Text:", height=150)

if st.button("Predict"):
    if user_input:
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)
        prob = model.predict_proba(X_input)[0].max()
        pred_label = label_map.get(prediction[0], str(prediction[0]))
        st.success(f"Prediction: {pred_label}")
        st.info(f"Confidence: {prob:.2f}")
    else:
        st.warning("Please enter some text.")


