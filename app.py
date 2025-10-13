import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import json
import os

BASE_MODEL = "distilbert-base-uncased"   # or your base model used in training
ADAPTER_PATH = "bbc_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    id2label_path = os.path.join(ADAPTER_PATH, "id2label.json")
    if os.path.exists(id2label_path):
        with open(id2label_path, "r") as f:
            id2label = json.load(f)
        labels = [id2label[str(i)] for i in range(len(id2label))]
    else:
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
