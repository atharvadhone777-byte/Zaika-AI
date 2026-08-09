"""
Builds the knowledge base the RAG assistant retrieves over.

Two sources, deliberately kept separate:
  1. A curated substitution/technique knowledge base - hand-written,
     because questions like "can I replace butter with oil" have answers
     that are genuinely general cooking knowledge, not something that can
     be mined from this specific recipe dataset. Writing ~20 good entries
     by hand is more reliable than trying to scrape or hallucinate this.
  2. Recipe-derived chunks - one short text chunk per recipe, generated
     from its own structured data (ingredients, steps, tags), so the
     assistant can answer recipe-specific questions ("is THIS recipe
     vegan-friendly") grounded in the actual recipe, not just general
     knowledge.

Both are embedded into the SAME retrieval index (see vector_store.py) so
a single retrieval call can surface whichever is more relevant to a given
question - the chatbot doesn't need to decide in advance which source a
question belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Hand-curated. Each entry is deliberately short and specific - RAG works
# best retrieving focused chunks, not long documents where the relevant
# sentence is buried among irrelevant ones.
SUBSTITUTION_KB: list[dict] = [
    {"id": "sub_butter_oil", "topic": "butter substitute substitution replacement alternative",
     "content": "Butter can be replaced with an equal amount of neutral oil (vegetable, canola, or light olive oil) in most savory cooking. For baking, use 3/4 the amount of oil as the butter called for, since butter also contains water that oil doesn't."},
    {"id": "sub_butter_vegan", "topic": "vegan butter substitute substitution replacement alternative",
     "content": "For a vegan substitute, use plant-based margarine or coconut oil in a 1:1 ratio for butter. Coconut oil works best in baking; olive oil is better for sauteing."},
    {"id": "sub_egg_baking", "topic": "egg substitute substitution replacement alternative baking",
     "content": "One egg in baking can be replaced with 1 tablespoon ground flaxseed mixed with 3 tablespoons water (let sit 5 minutes to gel), or with 1/4 cup unsweetened applesauce for a milder flavor."},
    {"id": "sub_dairy_milk", "topic": "milk substitute substitution replacement alternative",
     "content": "Dairy milk can be replaced 1:1 with oat milk, soy milk, or almond milk in most recipes. Oat milk behaves closest to dairy milk in sauces due to its similar thickness."},
    {"id": "sub_cream", "topic": "heavy cream substitute substitution replacement alternative",
     "content": "Heavy cream can be replaced with full-fat coconut milk (the solid part from a chilled can) for a dairy-free option, or with a mix of milk and cornstarch for a lighter, lower-fat sauce."},
    {"id": "sub_onion", "topic": "onion substitute substitution replacement alternative",
     "content": "If you're out of onion, shallots or leeks make the closest substitute in flavor and texture. In a pinch, onion powder (1 teaspoon per medium onion) works for flavor, though it won't add texture."},
    {"id": "sub_garlic", "topic": "garlic substitute substitution replacement alternative",
     "content": "Garlic powder can replace fresh garlic at roughly 1/8 teaspoon powder per clove. It lacks the sharpness of fresh garlic, so consider using slightly more if the dish relies on strong garlic flavor."},
    {"id": "sub_soy_sauce", "topic": "soy sauce substitute substitution replacement alternative",
     "content": "Soy sauce can be replaced with tamari (naturally gluten-free) 1:1, or with coconut aminos for a lower-sodium, slightly sweeter alternative."},
    {"id": "tech_air_fryer", "topic": "adapting recipes for an air fryer",
     "content": "Most oven recipes can be adapted for an air fryer by reducing the temperature by about 25°F (14°C) and checking for doneness at roughly 20% less time than the original recipe, since air fryers circulate heat more efficiently."},
    {"id": "diet_diabetic", "topic": "diabetic-friendly recipe adjustments",
     "content": "For a diabetic-friendly version of a recipe, reduce or replace added sugar with a low-glycemic sweetener, favor whole grains over refined ones, and pair carbohydrate-heavy dishes with a protein or fiber source to slow glucose absorption. This is general guidance, not medical advice - individuals should confirm dietary changes with their doctor or a registered dietitian."},
    {"id": "diet_lower_calorie", "topic": "reducing calories in a recipe",
     "content": "To reduce a recipe's calories, common levers are: use less oil/butter (or a light spray instead of pouring), substitute a portion of heavy cream with milk or broth, increase the vegetable-to-starch ratio, and choose leaner cuts of meat or reduce meat quantity in favor of legumes."},
    {"id": "diet_vegan_general", "topic": "making a recipe vegan",
     "content": "To make a typical recipe vegan: replace dairy milk/cream with a plant-based alternative, replace butter with plant margarine or oil, replace eggs using the substitutions above, and replace meat with a plant protein (tofu, tempeh, or legumes) suited to the dish's cooking method."},
    {"id": "tech_freezing", "topic": "freezing cooked meals",
     "content": "Most cooked grain, bean, and stew-based dishes freeze well for up to 3 months in an airtight container. Dishes with high dairy or cream content can separate when thawed - stir well while reheating gently to recombine."},
]


def build_recipe_chunks(recipe_df: pd.DataFrame) -> list[dict]:
    """
    Turns each recipe's structured fields into one short retrievable text
    chunk. This is template-based (not model-generated) on purpose - the
    underlying data is already structured and factual, so templating it
    into text is a faithful, hallucination-free representation of it. Using
    a generative model to write this description would risk introducing
    claims the structured data doesn't actually support.
    """
    chunks = []
    for _, row in recipe_df.iterrows():
        tags_str = ", ".join(row["tags"]) if row["tags"] else "no specific tags"
        content = (
            f"Recipe '{row['name']}' takes {row['minutes']} minutes and uses "
            f"{row['n_ingredients']} ingredients: {', '.join(row['ingredients'])}. "
            f"Tags: {tags_str}. Difficulty: {row['difficulty']}. "
            f"Nutrition: {round(row['calories'])} calories, {round(row['protein_pdv'])}% "
            f"daily value protein, {round(row['total_fat_pdv'])}% daily value fat, "
            f"{round(row['carbohydrates_pdv'])}% daily value carbohydrates."
        )
        chunks.append({"id": f"recipe_{row['id']}", "topic": row["name"], "content": content, "recipe_id": int(row["id"])})
    return chunks
