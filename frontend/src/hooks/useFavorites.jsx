import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "recipe-favorites";

export function useFavorites() {
  const [favorites, setFavorites] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(favorites));
  }, [favorites]);

  const toggleFavorite = useCallback((recipeId) => {
    setFavorites((prev) =>
      prev.includes(recipeId) ? prev.filter((id) => id !== recipeId) : [...prev, recipeId]
    );
  }, []);

  const isFavorite = useCallback((recipeId) => favorites.includes(recipeId), [favorites]);

  return { favorites, toggleFavorite, isFavorite };
}
