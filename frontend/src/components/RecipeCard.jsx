import { Clock, Flame, Heart } from "lucide-react";

const DIFFICULTY_COLOR = {
  easy: "text-sage",
  medium: "text-mustard",
  hard: "text-brick",
};

export default function RecipeCard({ recipe, onSelect, isFavorite, onToggleFavorite }) {
  const missingCount = recipe.missing_ingredients?.length ?? 0;

  return (
    <button
      onClick={() => onSelect(recipe.recipe_id)}
      className="group text-left bg-surface-raised rounded-card p-4 border border-white/5
                 hover:border-mustard/40 transition-colors relative"
    >
      <div className="flex justify-between items-start gap-2 mb-2">
        <h3 className="font-display text-lg leading-tight capitalize pr-6">{recipe.title}</h3>
        <span
          role="button"
          tabIndex={0}
          onClick={(e) => {
            e.stopPropagation();
            onToggleFavorite(recipe.recipe_id);
          }}
          className="absolute top-4 right-4 text-ink-muted hover:text-brick transition-colors"
          aria-label={isFavorite ? "Remove from favorites" : "Save to favorites"}
        >
          <Heart size={18} fill={isFavorite ? "currentColor" : "none"} className={isFavorite ? "text-brick" : ""} />
        </span>
      </div>

      <div className="flex items-center gap-4 font-mono text-xs text-ink-muted mb-3">
        <span className="inline-flex items-center gap-1">
          <Clock size={13} /> {recipe.cook_time_minutes} min
        </span>
        <span className={`inline-flex items-center gap-1 capitalize ${DIFFICULTY_COLOR[recipe.difficulty] || ""}`}>
          <Flame size={13} /> {recipe.difficulty}
        </span>
      </div>

      {missingCount === 0 ? (
        <span className="inline-block bg-sage-soft text-sage text-xs font-medium px-2.5 py-1 rounded-full">
          You have everything
        </span>
      ) : (
        <span className="inline-block bg-brick-soft text-brick text-xs font-medium px-2.5 py-1 rounded-full">
          {missingCount} missing
        </span>
      )}
    </button>
  );
}
