import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import os

MODEL_PATH = "bbc_model"

# Load model and tokenizer (cached for efficiency)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    # Load label mapping
    id2label_path = os.path.join(MODEL_PATH, "id2label.json")
    if os.path.exists(id2label_path):
        with open(id2label_path, "r") as f:
            id2label = json.load(f)
        labels = [id2label[str(i)] for i in range(len(id2label))]
    else:
        # fallback if no file exists
        labels = ["business", "entertainment", "sport", "tech", "politics"]
    return tokenizer, model, labels

tokenizer, model, labels = load_model()

st.title("BBC News Text Classifier 🗞️")
st.write("Enter a news article to predict its category.")

text = st.text_area("Enter text here")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_index = torch.argmax(probs, dim=-1).item()

        predicted_category = labels[pred_index]
        st.success(f"**Predicted Category:** {predicted_category}")
