import { useState } from "react";
import { Moon, Sun, ChefHat } from "lucide-react";
import { ThemeProvider, useTheme } from "./hooks/useTheme";
import { useFavorites } from "./hooks/useFavorites";
import Home from "./pages/Home";
import RecipeDetail from "./components/RecipeDetail";

function Header() {
  const { theme, toggleTheme } = useTheme();
  return (
    <header className="flex items-center justify-between mb-10">
      <div className="flex items-center gap-2">
        <ChefHat size={22} className="text-mustard" />
        <span className="font-display text-lg">Pantry</span>
      </div>
      <button
        onClick={toggleTheme}
        aria-label="Toggle dark mode"
        className="text-ink-muted hover:text-mustard transition-colors"
      >
        {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
      </button>
    </header>
  );
}

function AppContent() {
  const [selectedRecipeId, setSelectedRecipeId] = useState(null);
  const { favorites, toggleFavorite } = useFavorites();

  return (
    <div className="min-h-screen px-4 py-8 sm:px-8 max-w-4xl mx-auto">
      <Header />
      {selectedRecipeId ? (
        <RecipeDetail
          recipeId={selectedRecipeId}
          onBack={() => setSelectedRecipeId(null)}
          onSelectRecipe={setSelectedRecipeId}
          isFavorite={favorites.includes(selectedRecipeId)}
          onToggleFavorite={toggleFavorite}
        />
      ) : (
        <Home
          onSelectRecipe={setSelectedRecipeId}
          favorites={favorites}
          onToggleFavorite={toggleFavorite}
        />
      )}
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}
