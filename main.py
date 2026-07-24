import os
import json
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE_MODEL = "distilbert-base-uncased"
ADAPTER_PATH = "bbc_model"

tokenizer = None
model = None
labels = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, labels
    print(f"Loading base model '{BASE_MODEL}' and adapter '{ADAPTER_PATH}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=5,
        ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(device)
    model.eval()

    id2label_path = os.path.join(ADAPTER_PATH, "id2label.json")
    if os.path.exists(id2label_path):
        with open(id2label_path, "r") as f:
            id2label = json.load(f)
        labels = [id2label[str(i)] for i in range(len(id2label))]
    else:
        labels = ["business", "entertainment", "sport", "tech", "politics"]
    
    print(f"Model loaded successfully with labels: {labels}")
    yield

app = FastAPI(title="BBC News Classifier API", lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

@app.get("/api/health")
def health_check():
    return {"status": "ok", "device": str(device), "labels": labels}

@app.post("/api/predict")
def predict_category(request: PredictRequest):
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        pred_index = torch.argmax(probs, dim=-1).item()

    category_probs = {labels[i]: round(float(probs[i].item()), 4) for i in range(len(labels))}

    return {
        "predicted_category": labels[pred_index],
        "confidence": round(float(probs[pred_index].item()), 4),
        "probabilities": category_probs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
