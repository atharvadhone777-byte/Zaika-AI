# 🍳 Zaika — AI Recipe Generator

**Zaika** is an AI-powered recipe recommendation app that helps you decide **what to cook with the ingredients you already have.**

Just enter your ingredients, and Zaika:

* 🥘 Recommends matching recipes
* 🛒 Shows missing ingredients
* 🔄 Suggests ingredient substitutions
* 💬 Answers cooking-related questions using a RAG-based assistant

## ✨ Features

* 🤖 **Deep Learning Recipe Recommender**
* 🔍 **Ingredient-based Recipe Search**
* 💬 **RAG Cooking Assistant**
* 🔄 **Smart Ingredient Substitutions**
* ⚡ **FastAPI Backend**
* 🎨 **React Frontend**
* ❤️ **Favorites & Dark/Light Mode**
* 🧪 **22 Automated Tests**

## 🧠 How It Works

```text
Your Ingredients
       ↓
Deep Learning Model
       ↓
Recipe Retrieval
       ↓
Best Matching Recipes
       ↓
RAG Assistant → Cooking Help & Substitutions
```

The recommendation system uses a **Siamese Dual Encoder** trained with contrastive learning to match ingredients with suitable recipes.

The cooking assistant uses **RAG + TF-IDF retrieval** to provide answers grounded in the recipe knowledge base.

## 🛠️ Tech Stack

**ML:** TensorFlow, Keras, Scikit-learn, FAISS
**Backend:** FastAPI, Python
**Frontend:** React, Vite
**Database/Storage:** Parquet, JSON
**Testing:** Pytest
**Deployment:** Docker, Render, Vercel

## 📊 Model Performance

| Metric    | Score |
| --------- | ----: |
| Recall@1  | 66.7% |
| Recall@5  | 81.5% |
| Recall@10 | 92.6% |
| MRR       | 75.3% |

> ⚠️ Current scores are from a small sample dataset. The complete pipeline is designed to work with the full Food.com dataset.

## 🚀 Run Locally

### Backend

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API docs:

```text
http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:5173
```

### Run Tests

```bash
pytest tests/ -v
```

## 📁 Project Structure

```text
Zaika-AI/
├── app/          # FastAPI backend
├── ml/           # ML pipeline & model
├── rag/          # RAG cooking assistant
├── frontend/     # React frontend
├── tests/        # Automated tests
├── notebooks/    # EDA
└── docs/         # Reports & documentation
```

## 🔮 Future Improvements

* Train on the complete recipe dataset
* Add hard-negative mining
* Improve RAG with neural embeddings
* Improve recipe personalization
* Scale deployment for more users

---

### 🍴 Built with AI, designed for everyday cooking.

**Zaika — Tell us what's in your kitchen. We'll tell you what to cook.**
