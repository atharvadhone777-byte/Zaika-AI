import { useState } from "react";
import { Search, SlidersHorizontal } from "lucide-react";
import IngredientInput from "../components/IngredientInput";
import RecipeCard from "../components/RecipeCard";
import { api } from "../api/client";

export default function Home({ onSelectRecipe, favorites, onToggleFavorite }) {
  const [ingredients, setIngredients] = useState(["tomato", "onion", "garlic", "rice"]);
  const [maxMissing, setMaxMissing] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const search = async () => {
    if (ingredients.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.recommend(ingredients, {
        topK: 8,
        maxMissingIngredients: maxMissing === "" ? null : Number(maxMissing),
      });
      setResults(res.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="font-display text-4xl mb-1">What's in your kitchen?</h1>
        <p className="text-ink-muted">Add what you've got. We'll find what to cook.</p>
      </div>

      <div className="bg-surface-raised rounded-card p-5 mb-8 border border-white/5">
        <IngredientInput ingredients={ingredients} onChange={setIngredients} />

        <div className="flex items-center justify-between mt-5 pt-4 border-t border-white/10">
          <label className="flex items-center gap-2 text-sm text-ink-muted">
            <SlidersHorizontal size={15} />
            Max missing ingredients
            <input
              type="number"
              min="0"
              value={maxMissing}
              onChange={(e) => setMaxMissing(e.target.value)}
              placeholder="any"
              className="w-16 bg-surface-sunken border border-white/10 rounded px-2 py-1 text-ink
                         focus:outline-none focus:ring-2 focus:ring-mustard/60"
            />
          </label>

          <button
            onClick={search}
            disabled={loading || ingredients.length === 0}
            className="inline-flex items-center gap-2 bg-mustard text-surface font-medium px-5 py-2.5
                       rounded-card hover:opacity-90 disabled:opacity-40 transition-opacity"
          >
            <Search size={16} />
            {loading ? "Searching..." : "Find recipes"}
          </button>
        </div>
      </div>

      {error && <p className="text-brick mb-4">{error}</p>}

      {results && (
        <div>
          <p className="text-ink-muted text-sm mb-4">{results.length} recipes found</p>
          {results.length === 0 ? (
            <p className="text-ink-muted">
              Nothing matched closely enough. Try loosening the missing-ingredients limit, or add a
              couple more ingredients.
            </p>
          ) : (
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {results.map((r) => (
                <RecipeCard
                  key={r.recipe_id}
                  recipe={r}
                  onSelect={onSelectRecipe}
                  isFavorite={favorites.includes(r.recipe_id)}
                  onToggleFavorite={onToggleFavorite}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
