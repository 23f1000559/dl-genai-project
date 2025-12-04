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
            label = result["label"]
            score = result["score"]
            label_map = {
                "LABEL_0": "Anger",
                "LABEL_1": "Fear",
                "LABEL_2": "Joy",
                "LABEL_3": "Sadness",
                "LABEL_4": "Surprise",
                
            }
            display_label = label_map.get(label, f"Unknown Label ({label})")
            confidence = f"{score * 100:.2f}%"
            st.subheader("Prediction:")
            st.write(f"The text expresses **{display_label}** with **{confidence}** confidence.")
        except Exception as e:
            st.error(f"Error during inference: {e}")
    else:
        st.warning("Please enter some text.")
