# Interview Preparation — 100 Questions

Organized by category. Every answer references the actual code/decisions in
this repo — not generic textbook answers — so you can point to a specific
file or line while answering. Where useful, a follow-up/cross-question is
included to show how deep the interviewer might push.

---

## 1. ML Problem Formulation (Q1–10)

**Q1. Why frame this as retrieval instead of generation?**
Retrieval ranks existing candidates; generation produces new text. The
product's actual need — "which of my known recipes fits these ingredients"
— is a ranking problem, not a text-generation problem. Generation adds
hallucination risk (invented steps, wrong quantities) for no benefit here,
since the recipes already exist in the dataset. See blueprint §5.
*Follow-up: when would generation be the right call?* If the product needed
genuinely novel recipes not in any dataset (e.g. "invent a fusion dish"),
generation would be unavoidable — but that's a different product.

**Q2. Why a dual encoder instead of a classifier over recipe IDs?**
A classifier needs a fixed, closed label set decided at training time —
adding a new recipe would mean retraining the output layer. A dual encoder
embeds new recipes by just running them through the same encoder and adding
to the index; no retraining needed. This also generalizes to a live product
where recipes get added continuously.

**Q3. Why weight-shared rather than two separate towers?**
Both sides of the pair (a user's pantry, a recipe's ingredient list) are
the same modality — an unordered set of ingredient tokens from the same
vocabulary. Separate towers make sense when sides are genuinely different
modalities (e.g. DPR's short question vs. long passage). Sharing weights
here is both simpler and forces one consistent notion of "what ingredients
go together."

**Q4. Why not use a classic content-based similarity metric (Jaccard/cosine over one-hot ingredient vectors) instead of a learned embedding?**
Jaccard similarity treats every ingredient as equally distinct — it can't
learn that "scallion" and "green onion" are related, or that "basil" and
"parmesan" tend to co-occur (Italian cuisine) in a way that should
influence ranking. A learned embedding captures those relationships from
data; a fixed metric can't.

**Q5. Why in-batch contrastive learning instead of explicit negative mining?**
Explicit negative mining (deliberately selecting "hard" wrong recipes) is
its own subproject at this dataset size, and in-batch negatives come for
free from ordinary batching — every other item in the batch is a negative.
It gets harder (more informative) automatically as the embedding space
improves. See `ml/training/losses.py`.

**Q6. What's the actual training signal — where do labels come from, since there's no user click data yet?**
Queries are simulated by subsampling a recipe's own ingredient list (30–80%
of it); the positive is the full ingredient list. This teaches "ingredients
from the same recipe belong together" without needing query logs the
product doesn't have yet (pre-launch). See `ml/data/dataset.py:make_training_pairs`.

**Q7. What's the known gap in that training signal, and how is it handled?**
It never teaches substitution relationships (butter/margarine playing
similar roles) since substitutable ingredients never co-occur as
query/positive by construction. That's intentionally left to the RAG
assistant, which handles substitution via a curated knowledge base instead
of asking the embedding model to learn it implicitly.

**Q8. Why not use collaborative filtering given the dataset has user interactions?**
Collaborative filtering needs enough interaction density per user/item to
find patterns, and a cold-start recipe with zero interactions gets no
signal at all. Content-based (ingredient) retrieval works from day one for
every recipe, including brand new ones. Interaction data is listed as a
natural extension once volume justifies a hybrid approach (blueprint §4).

**Q9. Why is "difficulty" a rule-based heuristic instead of a learned model?**
There's no ground-truth difficulty label in the data — training a second
model would just be learning to reproduce a heuristic we'd have to invent
anyway to generate the label. A transparent rule is more honest about what
it is, and it's directly explainable to a user. See `preprocessing.py:estimate_difficulty`.

**Q10. Why exclude "seq2seq recipe generation" even as future work priority #1?**
It's a materially different (and harder/riskier) product surface — free
text generation risks producing plausible-sounding but wrong/unsafe
instructions ("simmer for -5 minutes"), and is much harder to evaluate
reliably. It's noted as an option but deliberately not prioritized above
lower-risk improvements like real-data retraining.

---

## 2. This Project's Specific Design Decisions (Q11–20)

**Q11. Why Food.com over RecipeNLG, given RecipeNLG is the "obvious" tutorial dataset?**
Food.com has nutrition and user-interaction data already present — both
required outputs of this product. RecipeNLG would need those stitched in
from elsewhere. See blueprint §4 for the full comparison table.

**Q12. Why does `ml/` never import FastAPI?**
It keeps training-time code testable and reasoned-about independently of
the web framework, and it's a hard signal of layering discipline —
swapping FastAPI for Flask would only touch `app/`, nothing in `ml/`.

**Q13. Walk me through what happens end-to-end from a `/recommend` request to a response.**
Route (`app/api/routes_recipe.py`) validates the Pydantic request, calls
`RecipeService.recommend`, which calls `RecipePredictor.recommend`
(`ml/inference/predictor.py`): normalizes ingredients, embeds them via the
loaded encoder, queries the FAISS index for `top_k * fetch_multiplier`
candidates, computes missing ingredients per candidate, filters by
`max_missing_ingredients`, attaches similar-recipe IDs, and returns.

**Q14. Why fetch `top_k * fetch_multiplier` candidates instead of exactly `top_k`?**
`max_missing_ingredients` filtering happens after retrieval. If exactly
`top_k` were fetched and then filtered, a strict missing-ingredient limit
could return fewer than `top_k` results even when good matches exist
slightly further down the ranking. Over-fetch-then-filter is simpler and
more correct than baking a hard constraint into the similarity search.

**Q15. Why does the service layer exist at all — why not call the predictor directly from the route?**
It keeps HTTP concerns (status codes, request/response shaping) separate
from business logic, and makes the business logic testable without
spinning up FastAPI. See `app/services/recipe_service.py`.

**Q16. Why is the encoder loaded once at startup instead of per-request?**
Loading a Keras model + FAISS index takes real, measurable time; doing it
per-request would make latency unpredictable for no benefit, since the
artifacts don't change between requests. See the `lifespan` handler in
`app/main.py`.

**Q17. What would break if two requests hit `/recommend` concurrently — is the predictor thread-safe?**
`RecipePredictor.recommend` doesn't mutate any shared state (no writes to
`self`), so concurrent reads are safe. TensorFlow's `predict()` and FAISS's
`search()` are both safe for concurrent read-only calls from async request
handlers in this single-process setup.

**Q18. Why is `missing_ingredients` a required (not optional) field in the response schema?**
Consistent response shape is deliberate: clients shouldn't need to branch
on "field present vs. absent." Even `/generate-recipe`, which has no user
ingredient context, returns an explicit empty list rather than omitting
the field. See `app/schemas/recipe_schemas.py`.

**Q19. Why POST for `/recommend` instead of GET, when it's conceptually a fetch?**
Ingredient lists are structured, potentially long arrays with optional
filters — GET query strings get unwieldy and are awkward to validate with
Pydantic. POST + JSON body is the correct RESTful choice despite
`/recommend` conceptually being a read.

**Q20. What's the actual data flow from raw CSV to a servable FAISS index?**
`clean_recipes` → `IngredientVocabulary.build` → `make_training_pairs` →
train the encoder (`train.py`) → export to H5 (`_export_encoder`) →
`build_index.py` encodes the FULL corpus and writes `recipe.index` +
`recipe_metadata.parquet` → `RecipePredictor` loads all of that at API
startup.

---

## 3. Deep Learning / Model Architecture (Q21–30)

**Q21. Why masked average pooling instead of an LSTM/GRU over the ingredient sequence?**
Ingredient lists have no meaningful order — `["tomato","onion"]` and
`["onion","tomato"]` are the same recipe. A sequence model would spend
parameters modeling order that carries no signal, and would need
positional encodings that actively mislead it.

**Q22. Why not a Transformer encoder for the ingredients?**
Same reasoning as Q21, plus: a Transformer's self-attention would be
solving a problem (modeling interactions between tokens in a *sequence*)
that doesn't match the data (an unordered *set*). Average pooling directly
matches what the data actually is, with far fewer parameters.

**Q23. Explain the masking implementation — why not just use Keras's `mask_zero=True`?**
`mask_zero=True` inserts automatic mask-propagation ops into the graph that
the legacy H5 saver couldn't reliably deserialize in this Keras 3 /
TensorFlow 2.16+ setup (`load_model` failed with an unresolvable-op error).
Masking is instead computed explicitly as ordinary layers (`Lambda` +
`Multiply`), which serializes cleanly to both SavedModel and H5.

**Q24. Why are the Lambda-layer functions registered with `@register_keras_serializable` instead of plain lambdas?**
Two failure modes with plain lambdas: (1) Keras 3 refuses to deserialize a
Lambda layer holding a bare lambda by default (arbitrary-code-execution
guard), and (2) even overriding that, a lambda's closure over module
globals doesn't reliably survive the marshal/unmarshal round trip, causing
`NameError` at load time. Named, registered functions fix both.

**Q25. Why `UnitNormalization` instead of a `Lambda` wrapping `tf.math.l2_normalize`?**
Same H5 serialization concern — `UnitNormalization` is a native Keras
layer that reloads without needing a custom-object scope, unlike a Lambda
wrapping a raw TF op.

**Q26. Why L2-normalize the embeddings at all?**
It makes dot product equivalent to cosine similarity, which is what the
contrastive loss optimizes against during training and what FAISS's
`IndexFlatIP` computes at serving time. Keeping that consistent avoids a
subtle train/serve skew.

**Q27. What's the parameter count and where does it come from?**
~90K parameters at `embedding_dim=128` (from the sweep results): mostly the
embedding table (`vocab_size × embedding_dim`) plus two dense layers
(256, 128 units) and the final projection. Small by design — the task
(set-to-vector embedding of a small closed vocabulary) doesn't need a large
model, and a smaller model trains faster and is easier to defend against
overfitting on a modest dataset.

**Q28. Why `Adam` and not `SGD` or `RMSprop`?**
Adam's adaptive per-parameter learning rates handle the sparse
gradient updates from an embedding table well (only rows for ingredients
present in a batch get non-trivial gradients), and it needs less manual
learning-rate tuning than SGD to get a reasonable first result — a
practical fit for a project on a tight time budget.

**Q29. Why symmetric loss (both query→positive and positive→query directions)?**
A query should retrieve its correct recipe, and a recipe's embedding
should be consistent with a plausible query — optimizing only one
direction leaves the embedding space asymmetric in a way that measurably
hurts representation quality, even though only the query→recipe direction
is used at serving time. CLIP uses the same symmetric construction.

**Q30. What does the temperature parameter in the contrastive loss control?**
It scales the logits before the softmax; lower temperature sharpens the
distribution, making the model more confident/punishing about near-miss
negatives. 0.07 is the CLIP/SimCLR-standard starting point, kept as-is
here since the sweep focused on embedding_dim and learning_rate instead
(documented as the two axes most likely to matter at this scale).

---

## 4. Training & Optimization (Q31–40)

**Q31. Walk me through what each of the four required callbacks does in this training loop.**
`EarlyStopping` halts once `val_loss` stalls and restores the best-epoch
weights (not the last, possibly-overfit epoch). `ModelCheckpoint` persists
the best epoch to disk independent of whether the run gets interrupted.
`ReduceLROnPlateau` shrinks the learning rate on plateau, typically
settling into a sharper minimum than a fixed LR. `TensorBoard` logs
per-epoch scalars. See `ml/training/train.py:build_callbacks`.

**Q32. Why `restore_best_weights=True` on EarlyStopping specifically?**
Without it, training stops at the best-monitored epoch but the model
object still holds the *last* epoch's weights, which is exactly the
epoch you were trying to avoid by stopping early.

**Q33. What does the hyperparameter sweep actually test, and why only those two axes?**
`embedding_dim` (model capacity) and `learning_rate` (optimization
stability) — the two hyperparameters most likely to matter for a model
this size. A wider sweep (dropout, hidden sizes, temperature) is scoped
out because each additional axis multiplies runtime, and this dataset is
too small for a fine-grained sweep to be meaningful anyway.

**Q34. What did the sweep actually find?**
`embedding_dim=128, lr=5e-4` gave the best val_loss (0.397) vs. `dim=64`
(0.446) and `dim=128, lr=1e-3` (0.407) — see `docs/hyperparameter_sweep.json`.
Larger embedding capacity helped; the slightly lower learning rate edged
out the higher one, though margins are small given the tiny sample dataset.

**Q35. Why `drop_remainder=True` only on the training set, not validation?**
The in-batch contrastive loss treats batch size as the number of negative
examples; a smaller final training batch would silently make that step's
loss "easier" (fewer negatives), skewing the training signal. Validation
doesn't have this issue since val_loss isn't used to update weights.

**Q36. How is reproducibility enforced across the pipeline?**
A single `RANDOM_SEED=42` (in `ml/config.py`) is reused for: numpy, the
train/val/test split, the query-subsampling in `make_training_pairs`, and
TensorFlow's global seed at the start of `train.py`. One seed defined once
means changing it happens in exactly one place.

**Q37. What would you do differently if training on the full 180K-recipe dataset instead of the 300-recipe sample?**
Re-check `min_ingredient_frequency` (the sample's vocab is artificially
small/flat, see EDA §3's caveat), widen the sweep grid now that more data
supports finer distinctions, and likely add hard-negative mining since
random in-batch negatives get "too easy" as the model improves on a larger,
more separable embedding space.

**Q38. Why is the training loop implemented via `model.fit()` with a custom loss, rather than a fully custom `train_step`/`GradientTape` loop?**
`model.fit()` gets all four required callbacks "for free" with their
standard, well-tested behavior. A custom loss function that receives the
full batch of `y_pred` is sufficient to implement in-batch contrastive
loss without needing a custom training loop — simpler and less
error-prone for the same result.

**Q39. What's `y_true` in the training data, and why is it a dummy?**
A zero array — unused by the loss function. The "label" for each row is
implicitly its own position in the batch (see `losses.py`'s `labels =
tf.range(batch_size)`), not an externally supplied target, so `y_true` is
just a placeholder Keras' `fit()` API requires.

**Q40. How long did training actually take, and does that matter for the interview story?**
Under 4 seconds per sweep run on this small sample (CPU only) — the model
is intentionally small. Worth stating plainly rather than overselling:
training time will scale up meaningfully on the real 180K-recipe dataset,
and that's an honest, expected next step, not a surprise.

---

## 5. Evaluation & Metrics (Q41–50)

**Q41. Why precision@k / recall@k / MRR instead of BLEU or ROUGE?**
BLEU/ROUGE measure n-gram overlap in generated *text* — this model ranks
candidates, it doesn't generate text. Reporting BLEU here would signal a
mismatch between what the model does and what's being measured. Ranking
metrics are what real recommendation/search systems use.

**Q42. With exactly one relevant item per query, isn't precision@k redundant with recall@k?**
Yes, with one relevant item precision@k is just recall@k / k — included
anyway because it's the standard metric name reviewers expect, and it
generalizes cleanly if "relevant" is ever broadened (e.g. "any recipe
sharing ≥80% of ingredients").

**Q43. Why is the retrieval corpus "all recipes" rather than just the test set?**
At serving time, a real user's pantry should retrieve from the ENTIRE
recipe catalog, not an artificially shrunk one. Evaluating against a
shrunk corpus makes the task artificially easier than production reality
— what's held out is the *queries* (test-set recipes' ingredient
subsets), not the corpus they're retrieved from.

**Q44. Explain what recall@1 = 0.667 actually means here.**
For 66.7% of test queries, the model's #1-ranked recipe was exactly the
recipe the query was sampled from. Given the query is often a substantial
subset of a real recipe's ingredients, this is a meaningful signal the
embedding space is capturing real ingredient-co-occurrence structure, not
just noise — though see the sample-size caveat (Q45).

**Q45. These numbers are on a 300-recipe sample — how much should I trust them?**
Directionally, they show the pipeline is learning real structure (recall
climbing from 67%→93% between k=1 and k=10, MRR of 0.75, is not what
you'd see from a broken or randomly-initialized model). As an estimate of
final production model quality, not much — the honest framing throughout
this project is "the pipeline works end-to-end," with final numbers
pending the real ~180K-row dataset.

**Q46. What's MRR measuring that recall@k doesn't?**
Recall@k is binary per query (found in top-k or not) and ignores *where*
in the top-k it was found. MRR (mean of 1/rank) rewards the correct item
being ranked higher within a hit, which recall@10 alone wouldn't
distinguish between rank 1 and rank 10.

**Q47. Why is FAISS's exact `IndexFlatIP` used for evaluation instead of an approximate index?**
The evaluation script does brute-force matrix multiplication directly
(`ml/evaluation/metrics.py`), which is exact by construction and instant
at this corpus size — no approximation error to account for when
interpreting the numbers. `ml/inference/retriever.py` uses FAISS for the
same exact computation at serving time, for consistency.

**Q48. How would you detect if the model were overfitting to the training set's specific recipes rather than learning generalizable ingredient relationships?**
Compare train vs. val loss trajectories (already logged via TensorBoard/
the sweep's `final_train_loss` vs `best_val_loss`) — a growing gap would
signal overfitting. Also: evaluate on genuinely held-out test queries
(already done) rather than reporting training-set retrieval performance,
which would be a much easier and less meaningful task.

**Q49. What qualitative check did you run beyond the aggregate numbers?**
`ml/evaluation/evaluate.py` prints a handful of actual query→top-3-results
examples with similarity scores, flagging which one was the "correct"
match — aggregate metrics can hide systematic failure patterns that
reading real examples surfaces immediately.

**Q50. If recall@1 were unexpectedly low after retraining on real data, what would you check first?**
In order: (1) the vocabulary — did `min_ingredient_frequency` filter out
too much signal at the larger scale; (2) the query-subsampling fractions
in `make_training_pairs` — too-small subsets make the task ambiguous; (3)
whether embedding_dim needs to scale up for the larger, more diverse
corpus; (4) a data leak or split bug (re-run the zero-overlap ID check
already in place for the splits).

---

## 6. NLP Fundamentals (Q51–60)

**Q51. Why a flat ingredient-level vocabulary instead of subword/BPE tokenization?**
Ingredients are closer to a closed categorical vocabulary (a few thousand
distinct normalized ingredients even at 180K recipes) than open-ended
natural language. Subword tokenization would add complexity (merge
tables, unknown-piece handling) without benefit — the model only needs to
know "was tomato present" and how it co-occurs with other whole
ingredients, not sub-token structure within "tomato."

**Q52. Where IS subword/wordpiece-style thinking relevant in this project?**
Nowhere currently, deliberately — the RAG assistant's TF-IDF vectorizer
operates on whole words/bigrams (`ngram_range=(1,2)`), which is a
different, simpler choice justified by the KB's short, keyword-dense text
(see Q71). If free-text recipe *generation* were added later, that's
where subword tokenization would become the right choice.

**Q53. Why `ast.literal_eval` instead of `eval()` when parsing the dataset's stringified lists?**
`literal_eval` only parses literal Python data structures (lists, dicts,
strings, numbers) and cannot execute arbitrary code, unlike `eval()`. This
matters because the input is scraped/aggregated data from an external
source — treating it as untrusted is the correct default, and the
codebase has a unit test (`test_parse_stringified_list_handles_malformed_input_safely`)
specifically confirming a code-injection-shaped string doesn't execute.

**Q54. Explain the ingredient normalization pipeline step by step.**
Lowercase → strip parenthetical asides ("onion (chopped)" → "onion") →
strip non-alphabetic characters (stray quantities/units) → collapse
whitespace → map known aliases to a canonical form (e.g. "green onions" →
"scallion"). Each step is conservative by design — merging obvious
duplicates without merging genuinely different ingredients.

**Q55. Why explicit alias mapping instead of a stemmer for ingredient normalization?**
A stemmer handles plurals well ("tomatoes"→"tomat") but can't fix
non-morphological aliases like "scallion" vs. "green onion" vs. "spring
onion," and risks mangling unrelated words. An explicit dict is more
precise for a closed, domain-specific vocabulary like cooking ingredients
— a genuine trade-off worth being able to defend (maintenance burden vs.
precision).

**Q56. Give an example of where over-aggressive normalization would be a bug, and how this pipeline avoids it.**
Collapsing "black pepper" into "pepper" would conflate it with "bell
pepper" or "chili pepper" — completely different ingredients. The
normalization is deliberately conservative and doesn't do this (there's a
unit test asserting `black pepper` stays `black pepper`, not `pepper`).

**Q57. What's the difference between what TF-IDF captures and what the trained embedding model captures?**
TF-IDF captures lexical/keyword overlap — it has no notion that "butter"
and "margarine" are related unless they co-occur in the same text. The
trained embedding model learns *distributional* relationships from
co-occurrence patterns across many recipes — it can place semantically
related ingredients near each other in vector space even without shared
words.

**Q58. Why does the RAG vector store weight the `topic` field by repeating it in the vectorized text?**
Found via manual testing: a query like "replace butter with oil" was
initially retrieving a longer, tangentially related chunk over the exact
"butter substitution" entry, because TF-IDF has more raw term-overlap
surface area in longer text. Repeating `topic` biases matching toward
chunks whose *topic*, not just incidental phrasing, matches the query.

**Q59. Describe a real bug you hit with word-form mismatches, and how you diagnosed and fixed it.**
The query "substitute butter" wasn't matching the KB entry topic "butter
substitution" — TF-IDF has no stemming, so "substitute" and "substitution"
are different tokens with zero shared vocabulary weight. Diagnosed by
testing retrieval directly against the vector store outside the API and
comparing scores. Fixed by broadening the KB topic keywords to include
both word forms rather than adding a stemming dependency for one bug.

**Q60. When would adding a proper stemmer/lemmatizer to the RAG retrieval be worth it over patching individual word-form mismatches?**
Once the KB grows large enough that manually anticipating every relevant
word form for every entry becomes impractical — at 13 curated entries,
patching specific mismatches found via testing was faster and more
precise; at, say, 200+ entries, a stemmer would systematically prevent
this whole class of bug instead of fixing them one at a time.

---

## 7. RAG & Retrieval (Q61–70)

**Q61. Why TF-IDF instead of a pretrained sentence-embedding model for the RAG assistant?**
A neural embedding model requires downloading pretrained weights over the
network at build/serve time — a real external dependency (availability,
size, licensing) for a knowledge base that's small and lexically dense
(substitution facts, not prose with subtle paraphrasing). TF-IDF captures
the great majority of relevant retrievals here without that dependency.
This was also a practical constraint in this environment (no access to
huggingface.co to download weights).

**Q62. Why extractive answer synthesis instead of feeding retrieved chunks to a generative LLM?**
Every word of an extractive answer is traceable to a specific KB entry —
zero hallucination risk, and the `source` field is always exactly
correct rather than a best-effort attribution. The trade-off: answers
read more like "here's the relevant fact" than a natural conversational
reply. `generate_answer()` is written as the one clearly-marked place to
plug in an LLM call later without changing anything upstream.

**Q63. How is confidence computed, and why that approach?**
Directly from the top retrieval score (>0.3 high, >0.12 medium, else low)
— an honest, direct function of how lexically close the question was to
what's actually in the KB, rather than a separately learned/guessed
number. It tells the caller exactly how trustworthy the extractive answer
is likely to be.

**Q64. Why does `/substitute-ingredient` restrict search to a specific `recipe_id` when given?**
Without it, similar-keyword recipe chunks (of which there are 259, one
per recipe) can outrank the correct KB entry purely by volume — with a
`recipe_id`, search restricts to that recipe's own chunk plus the always-
included curated KB, avoiding that noise. See `KnowledgeVectorStore.search`.

**Q65. What are recipe-derived chunks, and why are they template-generated rather than model-generated?**
One short text chunk per recipe (ingredients, cook time, tags, nutrition,
difficulty), built via string templating in `build_recipe_chunks`. Using
a generative model to write this description would risk introducing
claims the structured data doesn't actually support — templating is a
faithful, hallucination-free representation of data that's already
correct and structured.

**Q66. What happens when a question doesn't match anything in the KB well?**
`generate_answer` returns a low-confidence fallback message suggesting
what kinds of questions it can help with, rather than forcing a
low-quality answer from a barely-relevant chunk. Tested explicitly in
`test_chat_low_confidence_on_unrelated_question`.

**Q67. Why does the substitution KB include a disclaimer on the diabetic-adjustment entry specifically?**
Dietary/medical guidance carries real-world stakes beyond "the recipe
tastes different" — the entry explicitly states it's general guidance,
not medical advice, and suggests confirming with a doctor/dietitian. A
deliberate, not accidental, inclusion.

**Q68. How would retrieval quality change as the recipe corpus grows from 259 to 180,000 chunks?**
TF-IDF's core lexical matching would still function, but the "crowding"
effect seen in Q64 (many similar recipe chunks diluting the KB's IDF
weights) would get worse without the `recipe_id` scoping already in place.
This is exactly why that scoping exists rather than being an
afterthought — it was built anticipating this scaling behavior.

**Q69. What's the actual difference between `generate_answer` and `suggest_substitution`?**
`suggest_substitution` is a thin, more targeted wrapper: it constructs a
biased query ("substitute substitution replace {ingredient}") and returns
a narrower single-suggestion shape matching the `/substitute-ingredient`
schema, while `generate_answer` handles the general free-form `/chat`
case with a broader response.

**Q70. If you had one more day, what's the single highest-value RAG improvement?**
Swap in a real evaluation set — a list of (question, expected-KB-entry)
pairs — and compute retrieval precision the same rigorous way the main
model is evaluated, rather than relying on manual spot-checks. Currently
this is the weakest-evaluated component of the project, worth naming
honestly rather than glossing over.

---

## 8. FastAPI & Backend Engineering (Q71–80)

**Q71. Why are model artifacts loaded in a `lifespan` handler instead of at module import time?**
Importing `app.main` (e.g. for a quick route-listing script, or in test
setup) shouldn't force a multi-second TensorFlow + FAISS load as a side
effect of the import itself. `lifespan` ties loading to actual app
startup, not import.

**Q72. Walk me through the exception handling strategy.**
Domain-specific exceptions (`RecipeNotFoundError`, `ModelNotReadyError`)
map to specific status codes/error codes via registered handlers. A
catch-all handler logs the full traceback server-side but returns a
generic message to the client — internal error text never leaks into the
response body, while operators still get full detail in logs. See
`app/utils/exceptions.py`.

**Q73. Why Pydantic Settings instead of `os.environ` calls scattered through the code?**
Every config value is declared, typed, and validated in one place, with
environment-variable overrides "for free" — the standard 12-factor-app
approach, and directly relevant to the deployment's environment-variable
requirement. See `app/config/settings.py`.

**Q74. Why is CORS configured with an explicit origin allowlist rather than `"*"`?**
`"*"` disables a real browser security boundary (any site could call this
API from a user's browser) for no benefit once the frontend's actual
origin is known. Convenient in a demo, wrong for production — worth being
able to explain the difference, not just cite it as a rule.

**Q75. Why does `RecipeService` exist as a separate class rather than putting logic directly in the route function?**
Testability and separation of concerns: `RecipeService` can be unit
tested without spinning up FastAPI at all, and the route function stays
pure HTTP-shaping code (status codes, request parsing) rather than mixing
in business logic.

**Q76. How would you add authentication to this API without a major restructure?**
Add a dependency (FastAPI `Depends`) that validates a token/API key,
applied at the router level via `dependencies=[...]` on `include_router`
calls in `main.py` — the route/service/schema layering already in place
means auth is an additive middleware/dependency concern, not a rewrite.

**Q77. What's the actual test strategy here, and why real model artifacts instead of mocks?**
`tests/test_api.py` uses FastAPI's `TestClient` against the real trained
model and FAISS index (small sample data), not a mocked predictor. A
passing test means the whole stack — routes, services, ML inference, the
actual model — genuinely works together, which a mocked predictor
wouldn't verify.

**Q78. Why validate that `/recommend` results are returned in descending score order as part of the test suite, rather than just checking status codes?**
Status-code-only tests catch crashes but not silently-wrong behavior
(e.g. an accidental ascending sort would still return 200 OK). Asserting
actual response shape/ordering catches a class of bug status codes can't.

**Q79. What would happen under real concurrent load — is there a bottleneck?**
The Keras model's `.predict()` call is synchronous CPU-bound work inside
an async route handler — under real concurrent load this would block the
event loop per-request rather than truly running concurrently. The
documented next step (README's future improvements) is multi-worker
deployment; for single-worker/demo scale this wasn't yet a measured
problem, but it's the first place to profile if latency became an issue.

**Q80. Why is the health check's `model_loaded` field useful beyond a simple "is the process alive" check?**
A container can be running (process alive, port open) while still mid-way
through loading a multi-second model — a naive health check would report
healthy before the app can actually serve real traffic. Checking
`app.state.predictor is not None` distinguishes "process up" from
"actually ready," which matters for orchestrators (Render, k8s) deciding
when to route traffic.

---

## 9. Deployment & DevOps (Q81–90)

**Q90. Why Render for the backend instead of Railway, given both are mentioned as options?**
Render's tier behavior is more predictable for a FastAPI process holding a
loaded model in memory continuously; Railway's usage-based pricing is
harder to predict for a long-running portfolio project. Both are
defensible — the important thing in an interview is being able to justify
the actual choice, not that one is objectively "correct."

**Q82. Why is the model baked into the Docker image rather than downloaded at container start from object storage?**
At this model's size (a few MB), baking it in keeps deployment to a
single self-contained artifact with no extra download step or external
storage dependency to wire up. Explicitly documented as a trade-off that
should change if the model grows significantly larger — not a permanent
architectural stance.

**Q83. Explain the multi-stage Docker build for the backend.**
Stage 1 (`builder`) installs Python dependencies into a user site-packages
directory; stage 2 copies only that installed-packages directory plus
application code into a slim final image. This avoids shipping build
tooling in the runtime image, reducing size and attack surface.

**Q84. Why run the container as a non-root user?**
Basic hardening: if the app process is ever compromised, it doesn't run
with root privileges inside the container, limiting what an attacker
could do even in a worst-case scenario.

**Q85. Why is the frontend's API base URL baked in at build time (Vite env var) rather than configured at runtime?**
Vite resolves `import.meta.env.*` at build time, not runtime — this is a
Vite/static-site constraint, not a choice. It's handled by passing the
backend URL as a Docker build arg so the same Dockerfile works against
any backend URL without editing source, and by setting it as a Vercel
environment variable for that platform's build step.

**Q86. Why does `docker-compose.yml` include a healthcheck-based `depends_on` condition for the frontend?**
Without it, the frontend container could start and the browser could load
before the backend has finished loading the model, showing confusing
connection-refused errors. `condition: service_healthy` ensures the
frontend only starts after the backend's `/health` check passes.

**Q87. What wasn't verified in this environment, and why does that matter to disclose?**
Docker wasn't available in the development sandbox, so the Dockerfile was
carefully reviewed but never actually run through `docker build`; Render
and Vercel weren't reachable either, so the deployment steps are a
detailed best-effort guide, not a proven-working runbook. Stating this
explicitly (in `docs/DEPLOYMENT.md`) is itself good practice — silently
presenting untested steps as verified would be dishonest.

**Q88. If `/health` reported `model_loaded: false` after deploying, what's your debugging order?**
Check Render logs for the lifespan handler's loading step; confirm
`models/v1/` actually exists in the built image (not excluded by a
`.dockerignore`, not a path mismatch between local dev and the container's
`WORKDIR`); confirm the container has enough memory to load TensorFlow +
the model without being OOM-killed.

**Q89. Why single-worker uvicorn rather than multiple workers in the Dockerfile's CMD?**
Each worker process would duplicate the loaded model + FAISS index in its
own memory — at this project's scale, one worker handling requests via
FastAPI's async routes is sufficient, and adding workers is a documented,
deliberate future step once request volume actually justifies the extra
memory cost, not a default reached for without thought.

**Q90. How would you monitor this in production beyond the basic `/health` endpoint?**
Structured logging is already in place (`app/utils/logger.py`) as the
foundation; next steps would be request-level latency logging per
endpoint, a metrics endpoint (Prometheus-style) for FAISS query latency
and cache/model-load status, and alerting on `/health` failures — none of
which are built yet, worth stating honestly as a gap rather than implying
full observability exists.

---

## 10. Debugging Scenarios & Cross-Questions (Q91–100)

**Q91. Your model was failing to save as H5 with a "NotEqual" op error. Walk me through how you found and fixed it.**
The error traced to Keras's automatic mask-propagation machinery
(triggered by `mask_zero=True`) inserting an internal op the legacy H5
saver couldn't deserialize. Fixed by disabling automatic masking and
computing the pad mask explicitly via ordinary, serializable layers
(`Lambda` + `Multiply`) instead. Verified by actually reloading the saved
H5 file and re-running inference, not just confirming the save call
succeeded.

**Q92. After that fix, loading the H5 file threw `NameError: name 'tf' is not defined`. What was going on, and how is it different from the first bug?**
Different root cause, same symptom class (H5 + custom code): a plain
Python lambda's closure over module-level globals (`tf`) doesn't reliably
survive the marshal/unmarshal round trip used to persist Lambda layers.
Fixed by replacing lambdas with named, `@register_keras_serializable`
functions living in an always-imported module, not a marshalled closure.

**Q93. A `/generate-recipe` request returned a 500 with a Pydantic `ResponseValidationError` about a missing field. What happened, and what does that reveal about response validation?**
`get_recipe()` never set `missing_ingredients` (it has no user-ingredient
context to compute it against), and the schema required that field. This
is actually a *good* failure mode: FastAPI's response validation caught
a real inconsistency between what the predictor returned and what the
schema promised, as a clear 500 with a precise error, rather than silently
serving a malformed response to the frontend. Fixed by making the schema
field default to an empty list and having the predictor set it explicitly.

**Q94. The RAG assistant was confidently returning the wrong substitution for "replace butter with oil." How did you diagnose it wasn't a retrieval-ranking bug but a vocabulary bug?**
Tested the vector store directly (outside the API) with the raw query and
inspected which chunks scored highest and why — the "vegan" KB entry
literally contained the phrase "replace butter with oil" and out-scored
the dedicated "butter substitution" entry due to raw term overlap. That
ruled out a ranking-logic bug and pointed at a lexical/vocabulary mismatch
instead (see Q59).

**Q95. Your recommend endpoint returned zero results when `max_missing_ingredients=0` was set on the sample data. Was that a bug?**
No — verified by testing without the filter first: the synthetic sample
dataset pairs random unrelated ingredients per recipe (by construction,
see `make_sample_dataset.py`), so few recipes plausibly share many
ingredients with any query. Confirmed this was a data-realism limitation
of the synthetic sample, not an API or filtering logic bug, by checking
the underlying `missing_ingredients` counts directly.

**Q96. If precision@1 on the real 180K dataset came back much lower than on the sample, what would you check before assuming the model is bad?**
First, whether the vocabulary cutoff (`min_ingredient_frequency`) is
appropriately tuned for the real data's much longer ingredient tail (the
sample's distribution is flagged in the EDA notebook as too small to be
representative). Second, whether query-subsampling fractions still make
sense at scale. Only after ruling out data/config issues would I suspect
the architecture itself needs more capacity.

**Q97. A teammate suggests just using GPT-4/Claude directly for the whole recommendation task instead of this pipeline. How do you respond?**
For ranking a large, growing catalog against a user's ingredients, a
learned retrieval index is both cheaper (no per-request LLM call) and
faster (FAISS lookup vs. LLM latency) at scale, and its behavior is
directly evaluatable with precision/recall rather than relying on
subjective LLM output quality. An LLM is a better fit for exactly the
free-text, open-ended part of the product (the RAG assistant) — which is
why it's used there and not for ranking.

**Q98. How would you extend this system to support "I have most of these ingredients but want something different from what I usually cook"?**
This is a diversity/exploration-vs-relevance trade-off in ranking — could
be addressed by re-ranking retrieved candidates with a diversity penalty
(e.g. maximal marginal relevance) against the user's cooking history,
which isn't built yet but fits cleanly on top of the existing retrieval
step without changing the embedding model itself.

**Q99. What's the biggest thing you'd change about this project's scope if you had a full month instead of a week?**
Real interaction/rating data with enough volume to justify a genuine
hybrid recommendation approach (content + collaborative filtering), and a
proper offline A/B-style evaluation harness for the RAG assistant, which
is currently the least rigorously evaluated component (see Q70). Both
were explicitly scoped out for time, not overlooked.

**Q100. What's one decision in this project you're least confident about, and why?**
The RAG assistant's confidence scoring (Q63) is a reasonable heuristic but
untested against a real labeled evaluation set — it's plausible the
0.3/0.12 thresholds don't generalize well once the KB grows past its
current 13 hand-curated entries. Naming this honestly, rather than
defending it as more rigorous than it is, is itself part of a good
interview answer.
