# Deployment Guide

Backend → **Render**. Frontend → **Vercel**. Rationale for both choices is
in the project blueprint (`01_recipe_ai_project_blueprint.md`, §2);
this doc is just the "how", not the "why".

**Honesty note**: this sandbox environment has no Docker daemon and no
network access to Render/Vercel, so the steps below could not be executed
end-to-end here - the Dockerfile and configs were reviewed carefully but
not run through an actual `docker build` or live deploy. Treat this as a
detailed, best-effort guide to verify yourself on first deploy, not as
something that's been proven to work byte-for-byte.

## Backend → Render

1. Push this repo to GitHub.
2. On [render.com](https://render.com): **New > Web Service**, connect the repo.
3. Render auto-detects the `Dockerfile` at the repo root - confirm "Docker" is selected as the environment.
4. Settings:
   - **Root Directory**: leave blank (Dockerfile is at repo root)
   - **Instance type**: at least 1 GB RAM - TensorFlow + the loaded model need more headroom than Render's free 512 MB tier reliably provides
   - **Health check path**: `/health`
5. Environment variables (Render dashboard → Environment):
   ```
   APP_ENVIRONMENT=production
   APP_CORS_ALLOWED_ORIGINS=["https://<your-vercel-app>.vercel.app"]
   ```
6. Deploy. First build will be slow (installing TensorFlow); subsequent deploys reuse Docker layer caching and are faster.
7. Once live, note the Render URL (e.g. `https://ai-recipe-generator.onrender.com`) - the frontend needs it next.

**Model artifacts**: `models/v1/` is committed to this Docker image (via
`COPY models/v1/ ./models/v1/` in the Dockerfile) rather than downloaded
at container start. This is a deliberate size-vs-simplicity trade-off: for
a model this small (a few MB), baking it into the image keeps deployment
to a single artifact with no extra download step or external storage
dependency (S3, etc.) to wire up. If the model grows significantly larger
in a future iteration, switching to fetching it from object storage at
container startup is the documented next step - noted here rather than
built prematurely for a model that doesn't need it yet.

## Frontend → Vercel

1. On [vercel.com](https://vercel.com): **New Project**, import the same GitHub repo.
2. **Root Directory**: `frontend`
3. Framework preset: Vite (auto-detected)
4. Environment variable:
   ```
   VITE_API_BASE_URL=https://<your-render-backend>.onrender.com
   ```
5. Deploy. Vercel builds with `npm run build` and serves `frontend/dist` automatically - no Dockerfile needed here (the frontend's `Dockerfile`/`nginx.conf` in this repo are for the docker-compose local-dev path and for anyone who wants to self-host instead of using Vercel).
6. Once live, go back to Render and update `APP_CORS_ALLOWED_ORIGINS` to the real Vercel URL (step 5 under Backend), then redeploy the backend so CORS actually allows the deployed frontend to call it.

## Local development (no cloud accounts needed)

```bash
docker compose up --build
# backend:  http://localhost:8000
# frontend: http://localhost:3000
```

or without Docker:

```bash
# terminal 1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# terminal 2
cd frontend
npm install
echo "VITE_API_BASE_URL=http://localhost:8000" > .env
npm run dev
```

## Smoke-testing a deploy

```bash
curl https://<backend-url>/health
curl -X POST https://<backend-url>/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"ingredients": ["tomato", "onion", "garlic", "rice"], "top_k": 3}'
```

If `/health` returns `"model_loaded": false` after the startup window, check the Render logs for the model-loading step in `app/main.py`'s lifespan handler - the most likely cause is `models/v1/` not being present in the built image (confirm it wasn't excluded by `.dockerignore` if one is added later).
