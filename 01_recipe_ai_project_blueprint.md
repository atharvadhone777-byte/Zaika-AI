# AI Recipe Generator — Project Blueprint (Phase 1: Planning)

This is the planning deliverable you asked for. No code yet — just architecture, stack, dataset choice, ML formulation, API design, frontend wireframe, and a one-week roadmap, each with the reasoning behind it. Once you confirm, we move to implementation.

---

## 1. Project Architecture

### Why this shape

An interviewer at an ML org isn't just checking "does it work" — they're checking whether you understand **where ML ends and product engineering begins**. So the architecture deliberately separates three concerns that a lot of tutorial projects blur together:

1. **Offline ML pipeline** (data → training → artifacts) — runs once, not per-request.
2. **Online inference service** (FastAPI) — stateless, fast, loads pre-trained artifacts.
3. **Retrieval knowledge base** (for the RAG assistant) — separate from the trained model, so the chatbot can be updated without retraining anything.

This is exactly how production recommendation systems are structured (think Netflix, Spotify) — training is a batch job, serving is a thin, fast layer on top of frozen artifacts. It also means your API's p99 latency isn't hostage to model complexity, which is the first thing an interviewer will probe if you claim "production-quality."

### High-level flow

```
┌─────────────────────────┐
│   OFFLINE ML PIPELINE    │   (run in notebooks/scripts, not at request time)
│                          │
│  Raw Dataset             │
│     │                    │
│  Cleaning & Normalization│
│     │                    │
│  EDA                     │
│     │                    │
│  Train/Val/Test Split    │
│     │                    │
│  Ingredient Encoder      │
│  Training (Deep Model)   │
│     │                    │
│  Evaluation (P@k, R@k)   │
│     │                    │
│  Export: SavedModel +    │
│  Embeddings + FAISS Index│
└────────────┬─────────────┘
             │ artifacts (weights, vocab, embeddings, index)
             ▼
┌─────────────────────────┐        ┌───────────────────────┐
│     FastAPI Backend      │◄──────►│  Vector Store (FAISS)  │
│  api / services / ml /   │        │  Recipe embeddings +   │
│  schemas / utils / config│        │  RAG knowledge chunks  │
└────────────┬─────────────┘        └───────────────────────┘
             │ REST JSON
             ▼
┌─────────────────────────┐
│   Frontend (React/Next)  │
│  Ingredient input → UI   │
└─────────────────────────┘
```

### Why not a monolithic notebook-to-API script?
Alternative considered: put the model directly in the API process, train-on-the-fly or keep one big `app.py`. Rejected because (a) it makes the API cold-start slow, (b) it's untestable — you can't unit test ML logic separately from HTTP logic, (c) it signals "tutorial project" instantly to an interviewer. Separating `ml/` (pure Python, no FastAPI imports) from `api/` (routes only) means each layer can be tested and reasoned about independently — this is the single most common thing interviewers say separates a real engineer from a notebook user.

---

## 2. Technology Stack

| Layer | Choice | Why this over alternatives |
|---|---|---|
| DL framework | **TensorFlow / Keras** | You specified this; also Keras's `Model.save()` → SavedModel/H5 story is clean for the export requirement, and Keras callbacks (EarlyStopping, ReduceLROnPlateau) map directly onto your training requirements with minimal boilerplate. |
| Backend | **FastAPI** | Async-native, automatic OpenAPI docs (turns your "API documentation" README requirement into something auto-generated + curated), native Pydantic validation. Alternative: Flask — rejected because it needs extra libraries (Marshmallow, Flask-RESTX) to match what FastAPI gives natively, and async support is bolted-on rather than core. |
| Vector search | **FAISS** | Free, runs in-process (no extra infra to deploy on a free-tier host), fast enough for a dataset of ~100k–200k recipes. Alternative: Pinecone/Weaviate — rejected for MVP because they add a paid/hosted dependency for a project you need to demo cheaply and reliably; you can mention FAISS→Pinecone as a stated future improvement, which itself is a good interview talking point about scaling retrieval. |
| RAG generation | **Retrieval + a small open instruction model or extractive templating** (details in §5) | Keeps the "AI enhancement" grounded in your own retrieval index rather than being a thin wrapper around a hosted LLM, which is what the requirement is explicitly testing. |
| Data processing | **pandas, NumPy, NLTK/spaCy** for ingredient text normalization | Standard, well-understood, easy to justify line-by-line in an interview — no exotic library choices that you'd struggle to defend. |
| Experiment tracking | **TensorBoard** (as required) | Already mandated; also zero-cost to add via Keras callback. |
| Frontend | **React + Vite (or Next.js) + Tailwind** | Fast to build, deploys cleanly to Vercel, Tailwind keeps styling velocity high within a 1-week budget. Next.js only if you want SSR/SEO; for an authenticated demo app, plain Vite React is lighter and faster to ship — recommend **Vite** unless you specifically want SEO. |
| Backend hosting | **Render** (over Railway) | Render's free/hobby tier has more predictable cold-start behavior for FastAPI + a loaded Keras model; Railway is a fine alternative but usage-based pricing is less predictable for a portfolio project you'll leave running. Either is defensible — mention both, justify your pick. |
| Frontend hosting | **Vercel** | You specified this; zero-config for a Vite/Next app, generous free tier, instant preview URLs (useful to hand an interviewer a live link). |
| Containerization | **Docker** | Required for reproducible deployment; also lets you demonstrate you understand the difference between "works on my machine" and "works in production," which is a very common interview probe. |

---

## 3. Folder Structure

```
ai-recipe-generator/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
│
├── data/
│   ├── raw/                     # original downloaded dataset (gitignored)
│   ├── processed/               # cleaned/normalized data
│   └── splits/                  # train.csv / val.csv / test.csv
│
├── notebooks/
│   ├── 01_eda.ipynb             # exploratory data analysis only
│   └── 02_error_analysis.ipynb  # post-training qualitative review
│
├── ml/                          # pure ML code — no FastAPI imports allowed here
│   ├── __init__.py
│   ├── config.py                # hyperparameters, paths, seeds
│   ├── data/
│   │   ├── preprocessing.py     # cleaning, normalization
│   │   ├── tokenizer.py         # vocabulary + tokenization
│   │   └── dataset.py           # tf.data pipeline
│   ├── models/
│   │   ├── encoder.py           # ingredient/recipe encoder architecture
│   │   └── layers.py            # custom layers if any
│   ├── training/
│   │   ├── train.py             # training loop / Keras fit orchestration
│   │   └── callbacks.py         # EarlyStopping, checkpoint, LR scheduler configs
│   ├── evaluation/
│   │   ├── metrics.py           # precision@k, recall@k, BLEU/ROUGE if used
│   │   └── evaluate.py
│   └── inference/
│       ├── predictor.py         # loads SavedModel, exposes predict()
│       └── retriever.py         # FAISS index build + query
│
├── rag/
│   ├── knowledge_base.py        # builds chunked recipe knowledge docs
│   ├── vector_store.py          # FAISS wrapper for RAG
│   └── chatbot.py               # retrieval + answer synthesis
│
├── app/                         # FastAPI application — imports ml/ and rag/, nothing else does
│   ├── main.py
│   ├── api/
│   │   ├── routes_recipe.py     # /generate-recipe, /recommend
│   │   ├── routes_chat.py       # /substitute-ingredient, RAG chat
│   │   └── routes_health.py     # /health, /
│   ├── services/
│   │   ├── recipe_service.py    # business logic calling ml/inference
│   │   └── chat_service.py      # business logic calling rag/
│   ├── schemas/
│   │   ├── recipe_schemas.py    # Pydantic request/response models
│   │   └── chat_schemas.py
│   ├── utils/
│   │   ├── logger.py
│   │   └── exceptions.py
│   └── config/
│       └── settings.py          # env-based config (Pydantic Settings)
│
├── models/                      # exported artifacts (SavedModel, H5, FAISS index)
│   └── v1/
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_api.py
│   └── test_inference.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   └── api/                 # frontend's API client
│   ├── public/
│   └── package.json
│
└── docs/
    ├── architecture_diagram.png
    ├── api_examples.md
    └── postman_collection.json
```

**Rationale**: `ml/` never imports FastAPI; `app/` never contains training code. This one rule is what makes the codebase testable, and it's a natural place for an interviewer to probe ("what happens if you wanted to swap FastAPI for Flask?" — answer: you'd only touch `app/`, not `ml/`).

---

## 4. Dataset Selection

| Dataset | Size | Key features | Advantages | Disadvantages | Preprocessing challenges | Expected performance |
|---|---|---|---|---|---|---|
| **RecipeNLG** | ~2.23M recipes | title, ingredients, directions, NER-tagged ingredient list, source | Huge scale, ingredients already NER-extracted (saves work), good for generation research | Requires a data-request form to access; very noisy directions (scraped text); heavy compute to train on at full scale | Deduplication (many near-duplicate recipes from aggregator sites), directions need heavy cleaning | Good for large seq2seq generation, but overkill and slow to iterate on in a 1-week window |
| **Food.com Recipes and Interactions** (Kaggle) | ~180K recipes + ~700K user interactions/reviews | ingredients, steps, tags, nutrition (calories, fat, sugar, protein, etc.), techniques, user ratings | Nutrition info **already present** (removes an entire subsystem you'd otherwise have to build/estimate), interaction data enables genuine collaborative-filtering-style recommendation, manageable size for a laptop/Colab | Review text quality is inconsistent; ingredient strings are free-text and need normalization | Ingredient string parsing (quantities mixed with names), nutrition columns need unit sanity-checks | Strong — nutrition + interactions let you build recommendation AND satisfy the "nutritional information" output requirement without extra data engineering |
| **Epicurious Recipes (Kaggle)** | ~20K recipes | ingredients, ratings, nutrition, categories | Small, clean, fast to iterate | Too small for a deep model to learn rich embeddings; low diversity | Some missing nutrition fields | Fine for prototyping, weak for a "real" trained model to show off |
| **Kaggle "Recipe Ingredients Dataset" (Yummly)** | ~40K recipes | ingredient list, cuisine label | Very clean, great for classification | No instructions, no nutrition, no images — too narrow for the multi-output product you're building | Minimal — cleanest dataset of the five | Good only as a **secondary** dataset (e.g., cuisine classifier), not primary |
| **Hugging Face `recipe_nlg` / mirrors** | Same as RecipeNLG, HF-hosted | Same as RecipeNLG | Easier programmatic access via `datasets` library, no manual request form | Same noise issues as RecipeNLG | Same as RecipeNLG | Same as RecipeNLG |

### Recommendation: **Food.com Recipes and Interactions**

Reasoning:
- It's the only dataset in this list that gives you **ingredients + instructions + nutrition + user interactions** in one place — every one of your required outputs (recipe, missing ingredients, instructions, nutrition, "similar recipes" via interactions) is directly supported without stitching together multiple sources.
- ~180K recipes is large enough to train a real embedding model but small enough to preprocess and iterate on within a week on free-tier compute (Colab T4 or similar).
- The interaction data (user–recipe ratings) is what upgrades this from "text similarity toy" to an actual **recommendation system**, which is a materially stronger interview story than pure content-based matching.
- RecipeNLG is the "obvious" choice most tutorials reach for — using Food.com instead is itself a talking point ("I chose this because nutrition and interaction data let me support recommendation, not just generation, without needing three separate datasets").

---

## 5. ML Problem Formulation

| Approach | What it does | Implementation complexity | Explainability | Interview strength | Verdict |
|---|---|---|---|---|---|
| Multi-label ingredient prediction | Given a recipe, predict its ingredient tags | Low | High | Weak — doesn't solve the user's actual problem (input is ingredients, not recipes) | Rejected as primary task |
| **Ingredient-set → Recipe retrieval (dual-encoder / embedding model)** | Encode a user's ingredient set and every recipe into the same vector space; recommend recipes whose embedding is nearest | Medium | High — you can show nearest-neighbor reasoning, inspect embeddings, visualize clusters | Strong — touches representation learning, metric learning, retrieval, and is exactly how real recommendation systems at scale work | **Recommended as primary model** |
| Sequence-to-sequence recipe generation (generate full recipe text from ingredients) | Ingredients → generated instructions from scratch | High — needs large data, long training, careful decoding, prone to hallucinated/unsafe instructions ("simmer for -5 minutes") | Low — outputs are hard to verify or bound | Interesting to discuss, risky to demo live (generation quality on a 1-week budget with limited compute is often visibly bad) | Rejected as primary; mentioned as future work |
| Embedding-based retrieval + RAG for Q&A | Retrieval over a knowledge base to answer natural-language questions | Low–Medium (given retrieval infra already exists) | High | Strong, and it's your **optional AI enhancement** requirement, so it reuses infrastructure you're building anyway | **Recommended as the RAG assistant**, built on the same embedding infrastructure as the retrieval model |
| Transformer-based full generation (train a GPT-style model from scratch) | End-to-end generative model | Very high | Low | Practically infeasible in a week without a pretrained checkpoint, and even fine-tuning one adds real risk to your timeline | Rejected |

### Recommended formulation
**Primary model**: A **dual-encoder (Siamese) deep neural retrieval model** — one encoder for "ingredients-on-hand" (a bag/sequence of ingredient tokens), one encoder for "recipe" (ingredients + title, embedded the same way), trained so that a user's ingredient set lands close in embedding space to recipes they'd rate highly / that use those ingredients. At inference, retrieval is a nearest-neighbor search over precomputed recipe embeddings (via FAISS) — fast, explainable, and this *is* your "Deep Learning model" requirement, satisfied properly rather than superficially.

**Why this beats a raw seq2seq generator for your stated goals:**
- **Deep understanding**: forces you to reason about embedding spaces, contrastive/triplet loss, negative sampling, cosine vs. L2 similarity — all classic interview territory.
- **Clean implementation**: no beam search, no exposure-bias issues, no risk of the model producing nonsense instructions live in a demo.
- **Evaluatable properly**: precision@k / recall@k / MRR are clean, defensible metrics — unlike BLEU/ROUGE on recipe text, which are notoriously weak proxies for "is this a good recipe" (a good thing to say out loud in an interview — shows metric maturity).
- **Feasible in a week**: dual-encoder training on 180K recipes is realistically trainable on free-tier GPU in a few hours, leaving days for the API, frontend, RAG, and polish.

The **instructions, missing-ingredient list, nutrition, difficulty, cook time** for a retrieved recipe come directly from the dataset (with a lightweight rule-based difficulty estimator you can build and justify: e.g., a function of step count + total time + number of distinct techniques) — this is intentional: you don't need a second deep model to "generate" data that already exists in structured form. Building an unnecessary generator here would be the "unnecessary complexity" you said you want to avoid.

The **RAG assistant** (ingredient substitution, "make it vegan," "is this diabetic-friendly") is where free-text generation is actually justified, because those are genuinely open-ended questions that structured data can't answer — and it's grounded via retrieval over your recipe knowledge base plus a small curated substitution knowledge base (e.g., butter↔oil, dairy↔alternatives), so answers are traceable to source chunks rather than being pure LLM hallucination.

---

## 6. API Design

Base path: `/api/v1`

### `GET /health`
Liveness/readiness check. Returns model-loaded status, FAISS index status, uptime.
```json
{ "status": "ok", "model_loaded": true, "index_size": 178265, "uptime_seconds": 4213 }
```

### `GET /`
Basic service info + link to `/docs` (FastAPI's auto-generated OpenAPI UI).

### `POST /recommend`
Core recommendation endpoint — given ingredients on hand, return ranked recipe matches.

Request:
```json
{
  "ingredients": ["tomato", "onion", "garlic", "rice"],
  "top_k": 5,
  "max_missing_ingredients": 2,
  "dietary_filter": null
}
```
Response:
```json
{
  "results": [
    {
      "recipe_id": "10432",
      "title": "Spanish Tomato Rice",
      "match_score": 0.91,
      "required_ingredients": ["tomato", "onion", "garlic", "rice", "paprika"],
      "missing_ingredients": ["paprika"],
      "cook_time_minutes": 35,
      "difficulty": "easy",
      "nutrition": { "calories": 320, "protein_g": 6, "fat_g": 8, "carbs_g": 58 },
      "similar_recipe_ids": ["10891", "22045"]
    }
  ],
  "count": 5
}
```

### `POST /generate-recipe`
Given a `recipe_id` (usually chosen from `/recommend` results), return the full structured recipe — steps, timing per step if available, nutrition breakdown, difficulty.

### `POST /substitute-ingredient`
RAG-backed. Given a recipe context and a missing/unwanted ingredient, returns a grounded substitution suggestion with the source snippet it was derived from.

Request:
```json
{ "recipe_id": "10432", "ingredient": "butter", "constraint": "vegan" }
```
Response:
```json
{
  "suggestion": "Replace butter with an equal amount of olive oil or a plant-based margarine.",
  "confidence": "high",
  "source": "substitution_kb#dairy_fats"
}
```

### `POST /chat` *(RAG assistant, general Q&A)*
Free-form questions like "can I make this in an air fryer?" — same underlying retrieval pipeline as `/substitute-ingredient`, generalized.

### Design decisions worth defending
- **POST, not GET, for `/recommend`**: ingredient lists can be long and structured (arrays, filters) — GET query strings get unwieldy and awkward to validate with Pydantic; POST + JSON body is the correct RESTful choice here despite `/recommend` conceptually "fetching" data. Good to have this justification ready — it's a classic "why not GET" interview question.
- **Pydantic schemas for every request/response**: gives you automatic 422 validation errors for free, and doubles as your API documentation source of truth.
- **`missing_ingredients` returned even when zero**: consistent response shape is deliberate — clients shouldn't need to branch on "field present vs absent."
- **Errors**: structured error envelope (`{"error": {"code": "...", "message": "..."}}`) via a global FastAPI exception handler, not raw stack traces — a small thing that reliably signals production experience.

---

## 7. Frontend Wireframe

Single-page app, three primary views:

**1. Home / Ingredient Input**
- Ingredient chips with autocomplete (typeahead against your normalized ingredient vocabulary — reuses the same vocab your `ml/data/tokenizer.py` builds, so frontend and model never drift out of sync)
- "Find Recipes" CTA
- Optional filters: dietary constraint, max missing ingredients, max cook time

**2. Results / Recommendations**
- Card grid: title, match score, cook time, difficulty badge, missing-ingredients count
- Click-through to detail view

**3. Recipe Detail**
- Full ingredient list (owned vs. missing visually distinguished)
- Step-by-step instructions
- Nutrition panel
- "Similar recipes" carousel
- Embedded chat box for the RAG assistant ("Ask about this recipe")
- Save-to-favorites button
- Dark mode toggle (persisted in a settings context, not localStorage per the artifact constraint if this ever becomes a Claude artifact — for your actual deployed app, localStorage/a real backend user table is fine)

I'll render an actual visual wireframe and the system diagram below so you have something concrete to look at rather than just prose.

---

## 8. One-Week Implementation Roadmap

| Day | Focus | Deliverables |
|---|---|---|
| **Day 1** | Data + EDA | Download Food.com dataset, run `01_eda.ipynb`, document distributions (ingredient frequency, cook-time distribution, nutrition ranges, missing values), finalize cleaning rules |
| **Day 2** | Preprocessing + pipeline | `ml/data/preprocessing.py`, `tokenizer.py`, vocabulary build, train/val/test split with fixed seed, `tf.data` pipeline in `dataset.py`, unit tests for preprocessing |
| **Day 3** | Model + training | Dual-encoder architecture in `models/encoder.py`, training loop with EarlyStopping/ModelCheckpoint/ReduceLROnPlateau/TensorBoard, first training run, hyperparameter sweep (embedding dim, batch size, margin/temperature for contrastive loss) |
| **Day 4** | Evaluation + export | Precision@k, Recall@k, MRR on test split; qualitative error analysis notebook; export SavedModel + H5; build FAISS index over recipe embeddings; write `ml/inference/predictor.py` and `retriever.py` |
| **Day 5** | Backend | Full FastAPI app: routes, services, schemas, exception handling, logging; wire inference + retriever into `/recommend` and `/generate-recipe`; build substitution knowledge base + `rag/` module; `/substitute-ingredient` and `/chat` |
| **Day 6** | Frontend + integration | React app: input, results, detail views, chat widget, dark mode, mobile responsiveness; connect to live API; end-to-end manual testing |
| **Day 7** | Deployment + polish + docs | Dockerfile, deploy backend (Render), deploy frontend (Vercel), README (architecture diagram, API docs, examples, screenshots), Postman collection, final smoke tests |

This leaves you buffer inside each day rather than a literal 24-hour lockstep schedule — Day 3 (training) and Day 5 (backend) are the two days most likely to run long, so they're front-loaded relative to polish work.

---

## What I need from you to proceed

Once you confirm this plan (or tell me what to adjust — e.g., if you'd rather I pick RecipeNLG despite the tradeoffs, or you want Railway over Render), I'll start generating code **incrementally**, starting with the data pipeline, so you can run and verify each stage before we move to the next. I won't jump ahead to the 100 interview questions or full code until you've signed off here, per your instructions.
