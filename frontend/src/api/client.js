const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const message = body?.error?.message || `Request failed (${res.status})`;
    throw new Error(message);
  }
  return res.json();
}

export const api = {
  recommend: (ingredients, { topK = 5, maxMissingIngredients = null } = {}) =>
    request("/api/v1/recommend", {
      method: "POST",
      body: JSON.stringify({
        ingredients,
        top_k: topK,
        max_missing_ingredients: maxMissingIngredients,
      }),
    }),

  getRecipe: (recipeId) =>
    request("/api/v1/generate-recipe", {
      method: "POST",
      body: JSON.stringify({ recipe_id: recipeId }),
    }).then((res) => res.recipe),

  substituteIngredient: (ingredient, recipeId) =>
    request("/api/v1/substitute-ingredient", {
      method: "POST",
      body: JSON.stringify({ ingredient, recipe_id: recipeId }),
    }),

  chat: (question, recipeId) =>
    request("/api/v1/chat", {
      method: "POST",
      body: JSON.stringify({ question, recipe_id: recipeId }),
    }),
};
