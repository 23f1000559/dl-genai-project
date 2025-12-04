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
            raw_result = classifier(text)[0] 
            
            label = raw_result["label"]
            score = raw_result["score"]
            
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

            if display_label in ["Joy", "Surprise"]:
                st.success(f"The text expresses **{display_label}** with **{confidence}** confidence.")
            elif display_label in ["Anger", "Sadness", "Fear"]:
                st.error(f"The text expresses **{display_label}** with **{confidence}** confidence.")
            else:
                st.info(f"The text expresses **{display_label}** with **{confidence}** confidence.")
                
        except Exception as e:
            st.error(f"Error during inference: {e}")
    else:
        st.warning("Please enter some text.")
