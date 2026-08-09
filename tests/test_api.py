"""
API-level tests using FastAPI's TestClient. These exercise the app through
real HTTP requests against the real (small, sample-data) model artifacts -
not mocked - so a passing test here means the whole stack (routes ->
services -> ml/inference -> the actual trained model and FAISS index)
genuinely works together, not just that each piece works in isolation.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["index_size"] > 0


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_recommend_returns_ranked_results(client):
    r = client.post("/api/v1/recommend", json={"ingredients": ["tomato", "onion", "garlic", "rice"], "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] <= 3
    assert len(body["results"]) == body["count"]
    if body["results"]:
        scores = [res["match_score"] for res in body["results"]]
        assert scores == sorted(scores, reverse=True)  # ranked descending


def test_recommend_rejects_empty_ingredients(client):
    r = client.post("/api/v1/recommend", json={"ingredients": []})
    assert r.status_code == 422


def test_recommend_respects_max_missing_ingredients(client):
    r = client.post("/api/v1/recommend", json={
        "ingredients": ["tomato", "onion", "garlic", "rice"], "top_k": 10, "max_missing_ingredients": 0,
    })
    assert r.status_code == 200
    for res in r.json()["results"]:
        assert len(res["missing_ingredients"]) == 0


def test_generate_recipe_roundtrip(client):
    rec = client.post("/api/v1/recommend", json={"ingredients": ["tomato", "onion"], "top_k": 1}).json()
    recipe_id = rec["results"][0]["recipe_id"]

    r = client.post("/api/v1/generate-recipe", json={"recipe_id": recipe_id})
    assert r.status_code == 200
    assert r.json()["recipe"]["recipe_id"] == recipe_id


def test_generate_recipe_404_on_unknown_id(client):
    r = client.post("/api/v1/generate-recipe", json={"recipe_id": 999999999})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "recipe_not_found"


def test_substitute_ingredient(client):
    r = client.post("/api/v1/substitute-ingredient", json={"ingredient": "butter"})
    assert r.status_code == 200
    body = r.json()
    assert body["confidence"] in ("high", "medium", "low")
    assert "oil" in body["suggestion"].lower() or "margarine" in body["suggestion"].lower()


def test_chat_returns_grounded_answer_with_sources(client):
    r = client.post("/api/v1/chat", json={"question": "How can I make this recipe vegan?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert isinstance(body["sources"], list)


def test_chat_low_confidence_on_unrelated_question(client):
    r = client.post("/api/v1/chat", json={"question": "asdkjhasdkjhasd nonsense query xyz"})
    assert r.status_code == 200
    assert r.json()["confidence"] == "low"
