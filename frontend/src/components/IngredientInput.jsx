import { useMemo, useState } from "react";
import { X } from "lucide-react";

// A working autocomplete needs a vocabulary to suggest from. In production
// this would be fetched once from a lightweight backend endpoint (or
// bundled at build time from the same ml/data vocabulary the model uses,
// so frontend and model never drift apart - see the project blueprint's
// frontend wireframe notes). Kept as a local seed list here so the
// component works standalone without adding a new API endpoint.
const COMMON_INGREDIENTS = [
  "tomato", "onion", "garlic", "rice", "olive oil", "chicken breast", "butter",
  "flour", "egg", "milk", "basil", "parmesan cheese", "pasta", "ground beef",
  "bell pepper", "cumin", "paprika", "soy sauce", "ginger", "scallion", "lime",
  "cilantro", "black beans", "corn", "cheddar cheese", "sour cream", "potato",
  "carrot", "celery", "lemon", "mushroom", "spinach", "coconut milk",
  "chili powder", "avocado", "shrimp", "honey", "yogurt", "cucumber", "chickpeas",
];

export default function IngredientInput({ ingredients, onChange }) {
  const [draft, setDraft] = useState("");

  const suggestions = useMemo(() => {
    const q = draft.trim().toLowerCase();
    if (!q) return [];
    return COMMON_INGREDIENTS.filter(
      (i) => i.includes(q) && !ingredients.includes(i)
    ).slice(0, 6);
  }, [draft, ingredients]);

  const addIngredient = (value) => {
    const clean = value.trim().toLowerCase();
    if (clean && !ingredients.includes(clean)) {
      onChange([...ingredients, clean]);
    }
    setDraft("");
  };

  const removeIngredient = (value) => {
    onChange(ingredients.filter((i) => i !== value));
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      if (draft.trim()) addIngredient(draft);
    } else if (e.key === "Backspace" && !draft && ingredients.length > 0) {
      removeIngredient(ingredients[ingredients.length - 1]);
    }
  };

  return (
    <div>
      <div className="flex flex-wrap gap-2 mb-3">
        {ingredients.map((ing) => (
          <span
            key={ing}
            className="inline-flex items-center gap-1.5 bg-mustard-soft text-mustard px-3 py-1.5 rounded-full text-sm font-medium"
          >
            {ing}
            <button
              onClick={() => removeIngredient(ing)}
              aria-label={`Remove ${ing}`}
              className="hover:text-brick transition-colors"
            >
              <X size={14} />
            </button>
          </span>
        ))}
      </div>

      <div className="relative">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Add an ingredient and press Enter..."
          className="w-full bg-surface-sunken border border-white/10 rounded-card px-4 py-3
                     text-ink placeholder:text-ink-muted focus:outline-none focus:ring-2
                     focus:ring-mustard/60 transition-shadow"
        />
        {suggestions.length > 0 && (
          <ul className="absolute z-10 mt-1 w-full bg-surface-raised border border-white/10
                          rounded-card overflow-hidden shadow-xl">
            {suggestions.map((s) => (
              <li key={s}>
                <button
                  onClick={() => addIngredient(s)}
                  className="w-full text-left px-4 py-2 text-sm hover:bg-mustard-soft hover:text-mustard transition-colors"
                >
                  {s}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
