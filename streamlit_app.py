import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "m7k-run/dl-gen-ai-roberta" 

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

st.title("RoBERTa Text Classifier")
st.write("Enter your text.")

text = st.text_area("Input text:")

if st.button("Predict"):
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1).tolist()[0]

        st.subheader("Predictions:")
        for i, p in enumerate(probs):
            st.write(f"Class {i}: {p:.4f}")
    else:
        st.warning("Please enter some text.")
