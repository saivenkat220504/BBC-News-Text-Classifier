# BBC News AI Text Classifier 🗞️

A full-stack, Parameter-Efficient Fine-Tuned (**PEFT / LoRA**) text classification web application that categorizes news articles into 5 distinct topics (**Business**, **Entertainment**, **Politics**, **Sport**, and **Tech**). 

Powered by a fine-tuned **DistilBERT** model, a **FastAPI** REST API backend, and a modern, dynamic **React.js + Vite** frontend.

---

## 🌟 Key Features

- **Fine-Tuned NLP Model**: Utilizes `distilbert-base-uncased` enhanced with Low-Rank Adaptation (**LoRA**) for parameter-efficient inference.
- **Dynamic React.js Frontend**: Sleek, glassmorphic UI with real-time text input, 1-click sample article loaders, character counters, and smooth animations.
- **Model Confidence Distribution**: Displays real-time softmax probability percentage bars across all 5 BBC categories.
- **FastAPI Inference Service**: High-performance backend providing asynchronous prediction endpoints with built-in CORS support.
- **Dual UI Support**: Supports both the modern React.js frontend and the legacy Streamlit UI.

---

## 🛠️ Tech Stack

| Layer | Technology | Description |
|---|---|---|
| **Frontend** | React.js (Vite), Lucide Icons, CSS3 | Dynamic glassmorphic web interface |
| **Backend** | FastAPI, Uvicorn, Pydantic | High-speed REST API inference server |
| **Machine Learning** | PyTorch, Hugging Face Transformers, PEFT | Fine-tuned DistilBERT with LoRA adapters |
| **Dataset** | BBC News Dataset | 5 target categories (`business`, `entertainment`, `politics`, `sport`, `tech`) |

---

## 📁 Project Structure

```text
BBC-News-Text-Classifier/
├── bbc_model/              # Fine-tuned LoRA model weights & tokenizer configs
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── id2label.json
│   └── tokenizer.json
├── frontend/               # Dynamic React.js + Vite Frontend
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── index.css       # Dark glassmorphic design system
│   │   └── main.jsx
│   ├── index.html
│   └── package.json
├── main.py                 # FastAPI backend server
├── app.py                  # Streamlit legacy frontend
├── bbc-text.csv            # BBC News training dataset
├── requirements.txt        # Python package dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** installed
- **Node.js 18+** & **npm** installed
- **Git**

---

### 1. Clone the Repository

```bash
git clone https://github.com/saivenkat220504/BBC-News-Text-Classifier.git
cd BBC-News-Text-Classifier
```

---

### 2. Backend Setup (FastAPI)

1. **Create and activate a virtual environment**:

   **On Windows (PowerShell):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   **On Linux/macOS:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI backend server**:
   ```bash
   python main.py
   ```
   > Backend server runs on **`http://127.0.0.1:8000`**  
   > Swagger API Docs available at **`http://127.0.0.1:8000/docs`**

---

### 3. Frontend Setup (React.js)

1. Open a **new terminal tab/window** and navigate to `frontend`:
   ```bash
   cd frontend
   ```

2. **Install Node dependencies**:
   ```bash
   npm install
   ```

3. **Start the React development server**:
   ```bash
   npm run dev
   ```
   > Web app available at **`http://localhost:5173`**

---

## 📡 API Reference

### `GET /api/health`
Returns backend health status, compute device (`cpu`/`cuda`), and target labels.

**Response:**
```json
{
  "status": "ok",
  "device": "cpu",
  "labels": ["tech", "business", "sport", "entertainment", "politics"]
}
```

### `POST /api/predict`
Accepts article text and returns predicted category along with softmax probability distribution.

**Request Body:**
```json
{
  "text": "Quantum computing breakthrough allows processors to simulate complex chemical reactions..."
}
```

**Response:**
```json
{
  "predicted_category": "tech",
  "confidence": 0.648,
  "probabilities": {
    "tech": 0.648,
    "business": 0.141,
    "entertainment": 0.116,
    "sport": 0.048,
    "politics": 0.046
  }
}
```

---

## 💡 Running the Legacy Streamlit App

If you prefer to use the Streamlit interface:

```bash
python -m streamlit run app.py
```
> Runs at **`http://localhost:8501`**

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
