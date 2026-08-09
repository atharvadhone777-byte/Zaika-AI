import { useEffect, useState } from "react";
import { ArrowLeft, Clock, Flame, Heart } from "lucide-react";
import { api } from "../api/client";
import ChatWidget from "./ChatWidget";

export default function RecipeDetail({ recipeId, onBack, onSelectRecipe, isFavorite, onToggleFavorite }) {
  const [recipe, setRecipe] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setRecipe(null);
    setError(null);
    api.getRecipe(recipeId).then(setRecipe).catch((e) => setError(e.message));
  }, [recipeId]);

  if (error) {
    return (
      <div className="text-brick">
        Couldn't load this recipe: {error}.{" "}
        <button onClick={onBack} className="underline">
          Go back
        </button>
      </div>
    );
  }

  if (!recipe) {
    return <div className="text-ink-muted animate-pulse">Loading recipe...</div>;
  }

  return (
    <div>
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 text-sm text-ink-muted hover:text-mustard mb-4 transition-colors"
      >
        <ArrowLeft size={16} /> Back to results
      </button>

      {/* Signature element: the punch-hole card edge, see index.css .punch-holes */}
      <div className="punch-holes relative bg-surface-raised rounded-card pl-12 pr-6 py-6 border border-white/5">
        <div className="flex justify-between items-start gap-4 mb-3">
          <h1 className="font-display text-3xl leading-tight capitalize">{recipe.title}</h1>
          <button
            onClick={() => onToggleFavorite(recipe.recipe_id)}
            aria-label={isFavorite ? "Remove from favorites" : "Save to favorites"}
            className="shrink-0 text-ink-muted hover:text-brick transition-colors mt-1"
          >
            <Heart size={22} fill={isFavorite ? "currentColor" : "none"} className={isFavorite ? "text-brick" : ""} />
          </button>
        </div>

        {recipe.description && <p className="text-ink-muted mb-4">{recipe.description}</p>}

        <div className="flex items-center gap-5 font-mono text-sm text-ink-muted mb-6">
          <span className="inline-flex items-center gap-1.5">
            <Clock size={15} /> {recipe.cook_time_minutes} min
          </span>
          <span className="inline-flex items-center gap-1.5 capitalize">
            <Flame size={15} /> {recipe.difficulty}
          </span>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <section>
            <h2 className="font-display text-lg mb-2">Ingredients</h2>
            <ul className="space-y-1.5">
              {recipe.required_ingredients.map((ing) => (
                <li key={ing} className="text-sm capitalize flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-mustard shrink-0" />
                  {ing}
                </li>
              ))}
            </ul>

            <h2 className="font-display text-lg mt-6 mb-2">Nutrition</h2>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 font-mono text-sm text-ink-muted">
              <dt>Calories</dt>
              <dd className="text-ink text-right">{recipe.nutrition.calories}</dd>
              <dt>Protein (%DV)</dt>
              <dd className="text-ink text-right">{recipe.nutrition.protein_g}</dd>
              <dt>Fat (%DV)</dt>
              <dd className="text-ink text-right">{recipe.nutrition.fat_g}</dd>
              <dt>Carbs (%DV)</dt>
              <dd className="text-ink text-right">{recipe.nutrition.carbs_g}</dd>
            </dl>
          </section>

          <section>
            <h2 className="font-display text-lg mb-2">Instructions</h2>
            <ol className="space-y-3">
              {recipe.steps.map((step, i) => (
                <li key={i} className="text-sm flex gap-3">
                  <span className="font-mono text-mustard shrink-0">{String(i + 1).padStart(2, "0")}</span>
                  {step}
                </li>
              ))}
            </ol>
          </section>
        </div>

        {recipe.similar_recipe_ids?.length > 0 && (
          <section className="mt-8 pt-6 border-t border-white/10">
            <h2 className="font-display text-lg mb-3">Similar recipes</h2>
            <div className="flex gap-2 flex-wrap">
              {recipe.similar_recipe_ids.map((id) => (
                <button
                  key={id}
                  onClick={() => onSelectRecipe(id)}
                  className="text-sm bg-surface-sunken px-3 py-1.5 rounded-full hover:text-mustard transition-colors"
                >
                  Recipe #{id}
                </button>
              ))}
            </div>
          </section>
        )}

        <ChatWidget recipeId={recipe.recipe_id} />
      </div>
    </div>
  );
}
