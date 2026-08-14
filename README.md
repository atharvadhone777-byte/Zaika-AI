# 🍳 Zaika — AI Recipe Generator

**Zaika** is an AI-powered recipe recommendation system that helps you find recipes based on the ingredients you already have.

Enter what's available in your kitchen, and Zaika recommends suitable recipes, shows missing ingredients, suggests substitutions, and provides cooking assistance through a RAG-based chatbot.

---

## ✨ Project Overview

Zaika combines **Deep Learning, Recipe Retrieval, and RAG** to create a smart cooking assistant.

### Key Features

* 🤖 AI-based ingredient-to-recipe recommendation
* 🔍 Ranked recipe retrieval using a trained Deep Learning model
* 🛒 Identifies missing ingredients
* 🔄 Ingredient substitution suggestions
* 💬 RAG-based cooking assistant
* ⚡ FastAPI REST API
* 🎨 React frontend with Dark/Light mode
* ❤️ Recipe favorites
* 🧪 Automated tests

---

## 🧠 System Architecture

```text
                  User
                   │
                   ▼
            React Frontend
                   │
                   ▼
             FastAPI Backend
                   │
          ┌────────┴─────────┐
          ▼                  ▼
   Recipe Recommender     RAG Assistant
          │                  │
          ▼                  ▼
   Deep Learning Model   TF-IDF Retrieval
          │                  │
          ▼                  ▼
     FAISS Index       Knowledge Base
          │                  │
          └────────┬─────────┘
                   ▼
            Recipe Results
```

### ML Pipeline

```text
Recipe Dataset
      ↓
Data Cleaning & Preprocessing
      ↓
Train / Validation / Test Split
      ↓
Ingredient Vocabulary
      ↓
Siamese Dual Encoder
      ↓
Contrastive Learning
      ↓
Trained Model
      ↓
FAISS Recipe Index
      ↓
Recipe Recommendations
```

---

## 🛠️ Tech Stack

**Machine Learning:** TensorFlow, Keras, Scikit-learn, FAISS
**Backend:** Python, FastAPI
**Frontend:** React, Vite
**Testing:** Pytest
**Deployment:** Docker, Render, Vercel

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/atharvadhone777-byte/Zaika-AI.git
cd Zaika-AI
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

```bash
python ml/data/make_sample_dataset.py
```

For the complete dataset and preprocessing details, see:

```text
docs/DATASET.md
```

### 4. Prepare the ML pipeline

Clean the data, create the dataset splits, and build the ingredient vocabulary as described in the project documentation.

### 5. Train the model

```bash
python -m ml.training.train --sweep --epochs 30
```

### 6. Evaluate the model

```bash
python -m ml.evaluation.evaluate
```

### 7. Build the recipe index and RAG knowledge base

```bash
python -m ml.inference.build_index
python -m rag.build_knowledge_base
```

---

## 🚀 Run the Application

### Start the Backend

```bash
uvicorn app.main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

### Start the Frontend

Open another terminal:

```bash
cd frontend
npm install
```

Create `.env`:

```text
VITE_API_BASE_URL=http://localhost:8000
```

Then run:

```bash
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

## 🔌 API Documentation & Usage

Zaika provides REST APIs through FastAPI.

| Method | Endpoint                        | Description                   |
| ------ | ------------------------------- | ----------------------------- |
| `GET`  | `/health`                       | Check API and model status    |
| `POST` | `/api/v1/recommend`             | Get recipes from ingredients  |
| `POST` | `/api/v1/generate-recipe`       | Get complete recipe details   |
| `POST` | `/api/v1/substitute-ingredient` | Get ingredient substitutions  |
| `POST` | `/api/v1/chat`                  | Ask cooking-related questions |

### Example — Recipe Recommendation

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
-H "Content-Type: application/json" \
-d '{"ingredients":["tomato","onion","garlic","rice"],"top_k":3}'
```

For complete API examples, see:

```text
docs/postman_collection.json
docs/example_responses.json
```

You can also test all endpoints directly through:

```text
http://localhost:8000/docs
```

---

## 📊 Model Performance

The current model was evaluated on the sample dataset:

| Metric    | Score |
| --------- | ----: |
| Recall@1  | 66.7% |
| Recall@5  | 81.5% |
| Recall@10 | 92.6% |
| MRR       | 75.3% |

> **Note:** These results are from a small sample dataset. The complete pipeline is designed to work with the full recipe dataset.

---

## 🧪 Testing

Run the test suite with:

```bash
pytest tests/ -v
```

The project includes tests for preprocessing and the live FastAPI API.

---

## 📁 Project Structure

```text
Zaika-AI/
│
├── app/              # FastAPI backend
├── ml/               # ML pipeline and model
├── rag/              # RAG cooking assistant
├── frontend/         # React frontend
├── tests/             # Automated tests
├── notebooks/         # EDA and experiments
├── docs/              # Documentation and reports
├── data/              # Dataset files
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

* Train on the complete recipe dataset
* Improve recommendation quality with hard-negative mining
* Upgrade RAG retrieval with neural embeddings
* Add personalized recipe recommendations
* Scale the application for larger traffic

---

## 📄 License

MIT License