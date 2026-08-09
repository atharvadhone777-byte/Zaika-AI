# Pantry — AI Recipe Generator

An ingredient-aware recipe recommendation engine with a RAG-backed cooking assistant. Give it what's in your kitchen; it retrieves matching recipes ranked by a trained embedding model, tells you what's missing, and answers follow-up questions ("can I make this vegan?") grounded in a retrieval knowledge base.

Full design rationale for every decision below: [`01_recipe_ai_project_blueprint.md`](01_recipe_ai_project_blueprint.md).

## What's actually here

Every claim below was run and verified in this repo, not just planned:

- A **trained deep learning model** (TensorFlow/Keras, in-batch contrastive retrieval) — 67% recall@1, 93% recall@10 on the test split ([full report](docs/evaluation_report.json))
- A **hyperparameter sweep** with logged comparisons ([results](docs/hyperparameter_sweep.json))
- An **executed EDA notebook** with real plots ([notebook](notebooks/01_eda.ipynb))
- **22 passing tests** across preprocessing and the live API (`pytest tests/ -v`)
- A **FastAPI backend** with 5 REST endpoints, tested via `TestClient` against the real trained model
- A **RAG assistant** (TF-IDF retrieval, not a hosted LLM call) grounded in a curated substitution knowledge base + recipe-derived chunks
- A **React frontend** with dark/light mode, favorites, and an embedded chat widget — screenshotted in both themes during development

## Architecture

```
Offline ML pipeline (train once)          Online serving (every request)
─────────────────────────────             ──────────────────────────────
data/raw (CSV)                            FastAPI (app/)
  → ml/data/preprocessing.py                → app/api/  (routes)
  → ml/data/tokenizer.py (vocab)             → app/services/ (business logic)
  → ml/models/encoder.py (dual encoder)      → ml/inference/predictor.py
  → ml/training/train.py (+ sweep)             |- loads encoder.h5
  → ml/evaluation/evaluate.py                  |- FAISS index (retriever.py)
  → ml/inference/build_index.py                `- recipe metadata
      exports: encoder.h5, encoder_savedmodel/,
               recipe.index, recipe_metadata.parquet
                                           rag/ (RAG assistant)
                                             → TF-IDF vector store
                                             → substitution KB + recipe chunks
```

`ml/` never imports FastAPI. `app/` never contains training code. See the
blueprint §1 for why that boundary matters.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Data (ships with a schema-accurate sample; see docs/DATASET.md for the real ~180K-row dataset)
python ml/data/make_sample_dataset.py

# 2. Clean, split, build vocab (see notebooks/01_eda.ipynb to explore first)
python -c "
import pandas as pd, json
from ml.data.preprocessing import clean_recipes
from ml.data.dataset import split_dataset
from ml.data.tokenizer import IngredientVocabulary
from ml.config import *

df = pd.read_csv('data/raw/RAW_recipes.csv')
clean, report = clean_recipes(df)
print(report.summary())

out = clean.copy()
for c in ['ingredients','steps','tags']: out[c] = out[c].apply(json.dumps)
out.to_parquet(CLEANED_RECIPES_PATH, index=False)

train, val, test = split_dataset(clean)
for split_df, path in [(train,TRAIN_SPLIT_PATH),(val,VAL_SPLIT_PATH),(test,TEST_SPLIT_PATH)]:
    tmp = split_df.copy()
    for c in ['ingredients','steps','tags']: tmp[c] = tmp[c].apply(json.dumps)
    tmp.to_parquet(path, index=False)

vocab = IngredientVocabulary.build(clean['ingredients'].tolist(), min_frequency=PREPROCESSING.min_ingredient_frequency)
vocab.save(VOCAB_PATH)
print(f'vocab size: {len(vocab)}')
"

# 3. Train (with hyperparameter sweep) + evaluate + export
python -m ml.training.train --sweep --epochs 30
python -m ml.evaluation.evaluate

# 4. Build the FAISS index + RAG knowledge store
python -m ml.inference.build_index
python -m rag.build_knowledge_base

# 5. Run the backend
uvicorn app.main:app --reload --port 8000
# -> http://localhost:8000/docs for interactive API docs

# 6. Run the frontend (separate terminal)
cd frontend
npm install
echo "VITE_API_BASE_URL=http://localhost:8000" > .env
npm run dev
# -> http://localhost:5173
```

Or with Docker: `docker compose up --build` (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)).

```bash
pytest tests/ -v   # 22 tests, all real (no mocked model)
```

## API

Full examples with real captured responses: [`docs/postman_collection.json`](docs/postman_collection.json) (importable into Postman) and [`docs/example_responses.json`](docs/example_responses.json).

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + model/index status |
| `POST /api/v1/recommend` | ingredients in -> ranked recipes out |
| `POST /api/v1/generate-recipe` | full recipe detail by id |
| `POST /api/v1/substitute-ingredient` | grounded substitution suggestion |
| `POST /api/v1/chat` | free-form RAG Q&A about a recipe or cooking in general |

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["tomato", "onion", "garlic", "rice"], "top_k": 3}'
```

## Model

**Architecture**: a weight-shared (Siamese) dual encoder — ingredients and recipes are the same modality (ingredient sets), so one encoder maps both into a shared embedding space, trained with in-batch InfoNCE contrastive loss. Full rationale, alternatives considered, and why this beats a from-scratch seq2seq generator for this problem: blueprint §5.

**Training**: `ml/training/train.py --sweep` runs a 3-config grid (embedding_dim × learning_rate), each with EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and TensorBoard logging. Results: [`docs/hyperparameter_sweep.json`](docs/hyperparameter_sweep.json).

**Evaluation**: precision@k / recall@k / MRR (not BLEU/ROUGE — this is a ranking task, not a generation task; see blueprint §5 for why). Current numbers (small sample dataset — see caveat below):

| Metric | Value |
|---|---|
| Recall@1 | 0.667 |
| Recall@5 | 0.815 |
| Recall@10 | 0.926 |
| MRR | 0.753 |

**Known limitation**: these numbers are from the ~300-recipe synthetic sample (see `docs/DATASET.md`), not the real ~180K-row Food.com dataset — this sandbox couldn't reach kaggle.com. Every script here is written and tested to run unmodified against the real data; re-run `ml/training/train.py` and `ml/evaluation/evaluate.py` after swapping in the real CSVs and treat these numbers as "the pipeline works," not "the final model quality."

## RAG Assistant

Retrieval: TF-IDF + cosine similarity (scikit-learn) over a curated substitution knowledge base (13 hand-written entries covering butter/egg/dairy/onion/garlic/soy-sauce substitutions, air-fryer adaptation, vegan/diabetic/lower-calorie guidance) plus one auto-generated chunk per recipe. Synthesis is extractive, not generative — every answer is traceable to a specific source, with zero hallucination risk. Why TF-IDF instead of a neural embedding model, and why extractive instead of generative: `rag/vector_store.py` and `rag/chatbot.py` docstrings, also summarized in the blueprint.

## Project structure

See the blueprint §3 for the full annotated tree and the reasoning behind the `ml/` / `app/` / `rag/` split.

## Deployment

Backend -> Render, frontend -> Vercel. Step-by-step guide with an honesty note about what was and wasn't verified in this environment: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Future improvements

- Swap the sample dataset for the real ~180K-row Food.com data and re-run training/evaluation
- Replace `IndexFlatIP` (exact search) with an approximate index (HNSW/IVF) once corpus size actually makes exact search slow — not needed yet at this scale (see `ml/inference/retriever.py`)
- Swap TF-IDF retrieval in the RAG assistant for a neural sentence-embedding model once paraphrase-robustness matters more than the added model-download dependency
- Add hard-negative mining to contrastive training (currently in-batch negatives only) for a larger dataset where random negatives become too easy
- Multi-worker backend deployment once request volume justifies the extra memory cost of duplicating the loaded model per worker

## Interview preparation

[`docs/INTERVIEW_PREP.md`](docs/INTERVIEW_PREP.md) — 100 questions with answers covering ML fundamentals, this project's specific design decisions, NLP, deep learning, FastAPI, and deployment.

## License

MIT — see [LICENSE](LICENSE).
