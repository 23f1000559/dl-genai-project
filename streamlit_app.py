import streamlit as st
from transformers import pipeline

MODEL_NAME = "m7k-run/dl-gen-ai-roberta"

@st.cache_resource
def get_pipeline():
    
    classifier = pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME)
    return classifier

classifier = get_pipeline()

st.title("RoBERTa Text Classifier")
st.write("Enter your text below:")

text = st.text_area("Input text:")

if st.button("Predict"):
    if text.strip():
        try:
            result = classifier(text)
            st.subheader("Prediction:")
            st.write(result)
        except Exception as e:
            st.error(f"Error during inference: {e}")
    else:
        st.warning("Please enter some text.")
