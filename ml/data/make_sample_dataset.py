from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from ml.config import DATA_RAW_DIR, RANDOM_SEED  # noqa: E402

random.seed(RANDOM_SEED)
INGREDIENT_POOL = [
    "tomato", "onion", "garlic", "rice", "olive oil", "salt", "black pepper",
    "chicken breast", "butter", "flour", "egg", "milk", "sugar", "basil",
    "parmesan cheese", "pasta", "ground beef", "bell pepper", "cumin",
    "paprika", "soy sauce", "ginger", "scallion", "sesame oil", "lime",
    "cilantro", "black beans", "corn", "cheddar cheese", "sour cream",
    "potato", "carrot", "celery", "chicken stock", "bay leaf", "thyme",
    "lemon", "parsley", "white wine", "heavy cream", "mushroom", "spinach",
    "coconut milk", "curry powder", "chili powder", "avocado", "tortilla",
    "shrimp", "fish sauce", "brown sugar", "vinegar", "honey", "yogurt",
    "cucumber", "feta cheese", "olives", "pita bread", "chickpeas",
]
CUISINE_TAGS = [
    "italian", "mexican", "indian", "chinese", "thai", "mediterranean",
    "american", "french", "japanese", "vegetarian", "quick", "healthy",
]
TITLE_TEMPLATES = [
    "{a} and {b} rice bowl", "creamy {a} {b} pasta", "spicy {a} tacos",
    "roasted {a} with {b}", "{a} {b} stir fry", "classic {a} soup",
    "{a} stuffed peppers", "one pan {a} and {b}", "{a} curry",
    "grilled {a} with {b} sauce",
]


def _fake_steps(n_steps: int) -> list[str]:
    verbs = ["heat", "chop", "saute", "simmer", "season", "mix", "bake", "serve", "stir", "combine"]
    return [f"{random.choice(verbs).capitalize()} the ingredients, step {i + 1}." for i in range(n_steps)]


def _fake_nutrition() -> list[float]:
    return [
        round(random.uniform(120, 850), 1),
        round(random.uniform(2, 60), 1),
        round(random.uniform(1, 50), 1),
        round(random.uniform(2, 45), 1),
        round(random.uniform(3, 55), 1),
        round(random.uniform(1, 40), 1),
        round(random.uniform(3, 40), 1),
    ]


def generate_recipes(n_recipes: int = 300) -> pd.DataFrame:
    rows = []
    for recipe_id in range(1, n_recipes + 1):
        n_ing = random.randint(4, 12)
        ingredients = random.sample(INGREDIENT_POOL, n_ing)
        n_steps = random.randint(1, 12)  
        a, b = random.sample(ingredients, 2)
        title = random.choice(TITLE_TEMPLATES).format(a=a, b=b)
        minutes = random.choice([random.randint(-5, 0)] * 1 + [random.randint(5, 180)] * 20)  
        tags = random.sample(CUISINE_TAGS, k=random.randint(1, 4))

        rows.append({
            "id": recipe_id,
            "name": title,
            "minutes": minutes,
            "contributor_id": random.randint(1000, 9999),
            "submitted": f"2019-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "tags": str(tags),
            "nutrition": str(_fake_nutrition()),
            "n_steps": n_steps,
            "steps": str(_fake_steps(n_steps)),
            "description": f"A simple {title} you can make on a weeknight.",
            "ingredients": str(ingredients),
            "n_ingredients": n_ing,
        })

    df = pd.DataFrame(rows)
    dupes = df.sample(frac=0.03, random_state=RANDOM_SEED)
    df = pd.concat([df, dupes], ignore_index=True)
    null_idx = df.sample(frac=0.02, random_state=RANDOM_SEED).index
    df.loc[null_idx, "description"] = None
    return df


def generate_interactions(recipe_ids: list[int], n_interactions: int = 1200) -> pd.DataFrame:
    rows = []
    for i in range(n_interactions):
        rows.append({
            "user_id": random.randint(1, 250),
            "recipe_id": random.choice(recipe_ids),
            "date": f"2019-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "rating": random.choices([0, 1, 2, 3, 4, 5], weights=[2, 2, 3, 6, 15, 20])[0],
            "review": "Tasted great, will make again." if random.random() > 0.5 else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    recipes = generate_recipes(n_recipes=300)
    recipes.to_csv(DATA_RAW_DIR / "RAW_recipes.csv", index=False)

    interactions = generate_interactions(recipe_ids=recipes["id"].unique().tolist(), n_interactions=1200)
    interactions.to_csv(DATA_RAW_DIR / "RAW_interactions.csv", index=False)

    print(f"Wrote {len(recipes)} sample recipes -> {DATA_RAW_DIR / 'RAW_recipes.csv'}")
    print(f"Wrote {len(interactions)} sample interactions -> {DATA_RAW_DIR / 'RAW_interactions.csv'}")


if __name__ == "__main__":
    main()
