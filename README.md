# La Catrina: AI outfit planner (one-day prototype)

A small Streamlit app that runs locally on CPU and works in a phone browser. It can:

1. Add clothing photos to a local wardrobe. Uploading is the main method, and on a phone it opens the camera. A camera widget is optional and needs HTTPS or localhost.
2. Detect each item's category with **CLIP zero-shot** (`openai/clip-vit-base-patch32`). The categories are shirt, T-shirt, jeans, trousers, skirt, dress, jacket, footwear and accessory.
3. Extract the **3 dominant colours** with K-Means (scikit-learn) and name them from a palette of 25 readable colours.
4. Let you correct the category and main colour from dropdowns. Corrections are saved to disk.
5. Plan a **week of 7 outfits**: top + bottom (or a dress) + footwear, with an optional jacket and accessory. Each outfit gets a score and a one-line explanation.
6. Learn your taste from **Like / Dislike** ratings with a scikit-learn **Logistic Regression** model, blend it with compatibility (default 70% / 30%), explain every personalised ranking, and show before/after rankings and model evaluation in **My style preferences**.
7. Measure classification accuracy against your own `labels.csv` (`evaluate.py`), and evaluate the preference model (`evaluate.py --preferences`).

## Honest status

- **The compatibility score is a baseline.** It is not trained on outfit data (for example Polyvore) yet. It averages two parts:
  - the mean pairwise **CLIP image-embedding cosine similarity** of the pieces (visual similarity, not learned "goes-together-ness"), and
  - **hand-written HSV colour-harmony rules**: neutrals reward, analogous hues (within 30°) reward, complementary hues (150° or more apart) reward, and two saturated colours that fit none of these are penalised.
- Category detection is zero-shot CLIP with hand-written prompts. No fine-tuning has been done. Measure it on your own photos with `evaluate.py`. On 900 public product images: category **87.2%** and exact main-colour name **21.9%** (see [Public-dataset evaluation](#public-dataset-evaluation)). On the 10 labelled photos in this project, category was 10/10 (optimistic: the prompts were adjusted after seeing those photos) and colour was 4/10.
- Dominant colours use a simple heuristic, not segmentation:
  1. Drop pixels that match the photo's border colours.
  2. Run K-Means on the centre 70% of the photo.
  3. Merge clusters with the same colour name.

  Dim lighting and busy backgrounds still cause mistakes. In testing on real phone photos, light-wash jeans and white shoes in shade came out as "grey". That is why colour editing exists.
- Personalisation is off until you have rated at least 6 outfits, including at least 2 likes and 2 dislikes. Until then the app ranks on compatibility only. With a handful of ratings, the model's learned preferences and accuracy numbers are tentative. See [Personalisation](#personalisation-how-la-catrina-learns-your-taste).

## Run it (Windows PowerShell)

```powershell
cd "C:\Users\Rakesh Kumar\La-catrina"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app.py
```

The first step installs the small CPU-only build of PyTorch. If you skip it, `requirements.txt` installs the default PyTorch build, which is much larger.

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

The first start downloads the CLIP model (about 600 MB) into the Hugging Face cache. After that the app works offline.
Open http://localhost:8501 in a browser.

## Open it on your phone (same Wi-Fi)

1. Start the app so it listens on your network:
   ```powershell
   streamlit run app.py --server.address 0.0.0.0
   ```
2. Find your PC's IP address: run `ipconfig` and look for the "IPv4 Address" of your Wi-Fi adapter, for example `192.168.1.23`.
3. On the phone, open `http://192.168.1.23:8501`.
4. If the page doesn't load, allow Python through Windows Firewall on **private** networks. Windows usually asks the first time.

Tap **Browse files** on the phone to choose **Camera** or **Photo library**. The in-page camera widget (`st.camera_input`) usually does **not** work over plain `http://<ip>`, because browsers only allow camera access on HTTPS or localhost. Use the upload button instead.

## Using the app

- **Add**: pick one or more photos, then tap **Add to wardrobe**. Each photo is resized, classified and colour-analysed, then saved to `data/wardrobe/`. Re-uploading a photo you already added is skipped.
- **Wardrobe**: check each item's predicted category, confidence and colour swatches. Change the category and main colour from the dropdowns, then tap **Save**.
- **Week plan**: tap **Plan my week**. You need at least one top + one bottom (or a dress) and one pair of footwear. Otherwise the app tells you what is missing.
  - Rules: no identical outfit twice, and no piece on more than 3 days. If the wardrobe is too small, the app relaxes these limits and says so on screen.
  - A variety penalty steers reused pieces into different combinations.
  - Rate outfits with **Like / Dislike**. Ratings save immediately, and you can change one by tapping the other button or clear it by tapping the highlighted button again. Each outfit can have only one rating.
  - The model retrains on every change. Each card shows the outfit's updated score and rank; tap **Re-plan with my feedback** to rebuild the week. A table compares each day's score and rank (1 = best of all possible outfits) before and after, including where the previous outfit now ranks.
  - **My style preferences** (top of the tab) shows your like and dislike counts, whether the model is ready, what it has learned, the personalisation weight slider, and an evaluation panel.

All data lives in `data/`:

- `data/wardrobe/`: photos
- `data/wardrobe.json`: items, predictions, corrections and CLIP embeddings
- `data/feedback.json`: ratings

To start over, stop the app and delete the `data` folder.

## Personalisation: how La Catrina learns your taste

This is a hybrid recommender: a fixed **compatibility** score plus a **learned preference** score.

**1. Features (`planner.OutfitScorer.features`).** Each outfit becomes 42 named, interpretable numbers:

| Group | Features |
|---|---|
| Garment types | `cat:<category>`: 1 if the outfit contains it (includes optional jacket / accessory) |
| Colours | `colour:<name>`: 1 if a piece's main colour is that palette colour |
| Colour harmony | `harmony:<kind>`: share of colour pairs that are neutral / analogous / complementary / clash / mixed |
| Overall | `clip_similarity` (mean CLIP similarity of the pieces), `neutral_share` (share of neutral pieces), `n_colours` (distinct main colours) |

Features are recomputed from the current wardrobe whenever the model trains, so a colour or category correction also corrects past ratings.

**2. Model (`preference.py`).** It is a scikit-learn `Pipeline(StandardScaler, LogisticRegression(C=0.5, class_weight="balanced"))`, trained on all saved ratings (like = 1, dislike = 0).

- **Standardising** puts binary and continuous features on one scale, so the coefficients are comparable.
- **L2 regularisation** (`C=0.5`) keeps probabilities cautious with few ratings.
- **`class_weight="balanced"`** stops a user who mostly likes from producing a model that predicts "like" for everything.
- The model learns one weight per feature. The predicted probability is `P(like) = sigmoid(b + Σ wᵢ·zᵢ)`, where `zᵢ` is the standardised feature.

**3. When it is used.** The model is only trained once there are **≥ 6 ratings with ≥ 2 likes and ≥ 2 dislikes** (`MIN_TOTAL` and `MIN_PER_CLASS` in `preference.py`). Before that, or if all ratings are the same label, the app **falls back to the original compatibility ranking** and says so.

**4. Final score (`planner.py`).**

```
compatibility = 0.5 × mean CLIP similarity + 0.5 × (mean HSV harmony + 1) / 2
compat_norm   = compatibility rescaled to 0–1 over all possible outfits in this wardrobe
final         = 0.7 × compat_norm + 0.3 × P(like)        (model ready)
final         = compatibility                            (fallback: identical to the original app)
```

- **Rescaling.** Raw compatibility varies in a narrow band. Without rescaling, even a 0.3 preference weight almost completely replaced the compatibility order (Spearman 0.06 in testing).
- **Guard.** For outfits in the bottom 25% of compatibility, P(like) is capped at 0.5. Likes can push a poor match down but can't lift it.
- **Weight.** `PREF_WEIGHT = 0.3` in `planner.py` is the default, and the slider changes it from 0 to 0.5. 0 ignores ratings; higher values let ratings reorder more. On the simulated demo ratings, the Spearman correlation with the compatibility-only order was 1.00 at weight 0.1, 0.89 at 0.3, and 0.62 at 0.5.

**5. Weekly plan.** The planner keeps its existing rules:

- 7 days;
- no identical outfit twice;
- no piece more than 3 times (relaxed with a message for small wardrobes);
- a variety penalty;
- outfits you disliked are skipped unless the wardrobe is too small.

It ranks with the final score.

**6. Explanations.** Each personalised outfit shows its P(like), how many places it moved compared with the compatibility-only ranking, and the **main factor**. The main factor is the feature with the largest log-odds contribution (coefficient × standardised value) in the direction of the prediction, for example "you tend to like olive pieces". A factor is only cited if it moves the log-odds by at least 0.3. Otherwise the outfit says the ratings don't strongly favour or penalise it. **My style preferences** lists the largest learned weights. Nothing is shown until the model is ready.

**7. Evaluation (app panel, or `python evaluate.py --preferences`).**

- **Training metrics:** accuracy, precision, recall and F1 on the ratings the model learned from. These are labelled optimistic.
- **Cross-validated metrics:** stratified k-fold, with the scaler and model re-fitted inside every fold, so held-out ratings never leak into training. These need at least 3 likes and 3 dislikes.
- **Majority-class baseline accuracy.**
- **Spearman correlation** between the personalised and compatibility-only rankings.
- **Average rank** of liked and disliked outfits under each ranking, labelled as training data.

### Measured results (honest)

- **Your real ratings:** 1 rating so far, so the model is not ready and there are no results.
- **Simulated user** (tests and demo; *illustrative, not real behaviour*): a scripted rule "likes navy, dislikes red" or "likes the olive / maroon tops, dislikes the grey top", with 8 ratings. The cross-validated accuracy was 1.00 against a 0.50 baseline. This only shows that the pipeline can learn a perfectly consistent rule. Real taste is noisier, and 8 ratings say little about generalisation.
- **Spurious patterns:** in the simulated run, the model also picked up "you tend to like jeans", only because the liked outfits happened to contain jeans. Small data creates spurious patterns, so treat the learned preferences as tentative until there are many ratings.

### Tests

```powershell
python test_personalisation.py
```

The script runs 11 tests on a synthetic wardrobe in a temporary data folder (your `data/` is never touched):

- readiness and fallback;
- one-class ratings;
- learning and generalising to unseen outfits;
- the exact score formula and the guard;
- weight 0 means the original ranking;
- grounded explanations;
- 7 valid outfits with variety rules and disliked outfits excluded;
- small-wardrobe relaxation;
- no duplicate ratings, plus changing and clearing a rating;
- old duplicate feedback files;
- evaluation.

`LACATRINA_DATA_DIR=<folder>` points the app or scripts at a different data folder, for demos.

## Accuracy script

1. Put your photos in `data/wardrobe/`. They can be added through the app or copied there by hand.
2. Create `labels.csv` in the project folder, using `labels.example.csv` as a template:
   ```
   filename,category,main_colour
   white_tee.jpg,T-shirt,white
   ```
   - `category` must be one of: shirt, T-shirt, jeans, trousers, skirt, dress, jacket, footwear, accessory.
   - `main_colour` must be one of: black, charcoal, grey, light grey, white, cream, beige, khaki, tan, brown, navy, denim, blue, light blue, teal, green, olive, yellow, mustard, orange, red, maroon, pink, purple, lavender. You can also leave it blank.

   The script reads the photos directly from the folder. Photos copied in by hand are **not** in the app's wardrobe until you upload them in the app. The app then stores its own resized copy.
3. Run:
   ```powershell
   python evaluate.py
   ```
   The script prints:
   - a per-photo result
   - overall category accuracy
   - per-class precision, recall and F1
   - a confusion matrix
   - how often the K-Means main colour exactly matches your colour label

Options: `--labels path.csv` and `--photos folder`.

## Public-dataset evaluation

Ten personal photos are too few for a credible accuracy figure, so `evaluate_public.py` also tests the unchanged vision pipeline on a public dataset. It runs evaluation only; nothing is trained.

- **Dataset:** "Fashion Product Images (Small)", the Hugging Face mirror `ashraq/fashion-product-images-small` (44,072 product photos, 60 × 80 px, labelled with type and colour).
- **Sample:** a random sample (seed 0) of 100 images per category. Dataset types are mapped to our 9 categories; types with no clear match are excluded.

```powershell
python evaluate_public.py --download --per-class 100
```
```powershell
python evaluate_public.py --run
```

The download is about 1.3 MB into `eval_data/`, which is git-ignored. The run writes `eval_results/public_eval.md` and a CSV of predictions.

**Measured results** (900 images):

| What | Result | 95% CI |
|---|---|---|
| Category accuracy, current prompts | **87.2%** | 84.9–89.2% |
| Category accuracy, one generic prompt per class | 88.4% | 86.2–90.4% |
| Main colour, exact palette name (869 images with a matching label) | 21.9% | 19.2–24.7% |
| Main colour, same colour family | 36.6% | 33.5–39.8% |

- **Categories.** The weakest distinction is jeans vs trousers (jeans recall 0.63). The three prompt designs are statistically indistinguishable, so the extra "tank top / camisole" prompts gave no measurable gain here.
- **Colours.** Colour naming is the weakest part of the pipeline. Shaded black fabric extracts to about RGB (61, 59, 59) and is named "charcoal"; shaded white is named "light grey".

**Colour calibration experiment** (`python evaluate_public.py --calibrate`, full report in `eval_results/colour_calibration.md`):

1. The colour-labelled images were split 50/50 into development and test halves.
2. The reference colours were learned on the 434 development images.
3. On the 435 held-out test images, exact accuracy rose from **20.2% to 39.3%** and family accuracy from **36.3% to 55.9%** (McNemar p ≈ 3.5e-12).
4. On the 10 labelled phone photos, however, exact accuracy fell from **4/10 to 2/10**: product-photo lighting doesn't transfer to room-lit phone photos.

**The app therefore keeps the original swatches.** The learned references are saved in `eval_results/colour_calibration_candidate.json`; to try them, copy that file to `colour_calibration.json` in the project folder.

## Reproducibility

The following use seed 0:

- Python, NumPy and torch
- K-Means (`random_state=0`)
- cross-validation folds for the preference evaluation (`random_state=0`)

Logistic Regression (lbfgs) is deterministic, so the same feedback file always gives the same model.

## Files

| File | What it does |
|---|---|
| `app.py` | Streamlit UI (Add / Wardrobe / Week plan) |
| `.streamlit/config.toml` | Vintage film theme: cream `#F5E6C8`, kodak tan `#D4A574`, faded teal `#4A7B7C`, dusty rose `#E8B4B8`, sepia `#8B4513`, ink `#4E342E` (film grain, polaroid and VHS-hover effects are CSS at the top of `app.py`) |
| `vision.py` | CLIP loading, zero-shot category, embeddings, K-Means colours |
| `colours.py` | Colour palette, RGB→name, HSV harmony rules |
| `planner.py` | Outfit scoring, explanations, greedy week planner with variety rules |
| `preference.py` | scikit-learn preference model (StandardScaler + LogisticRegression), readiness rules, evaluation |
| `storage.py` | Local JSON + photo storage; one rating per outfit (change / clear supported) |
| `evaluate.py` | Category accuracy against `labels.csv`; `--preferences` evaluates the preference model |
| `evaluate_public.py` | Category and colour evaluation on a sample of a public fashion dataset |
| `test_personalisation.py` | 11 tests: readiness, learning, scoring formula, explanations, planner validity, persistence, evaluation |

## Roadmap (not built yet)

The full system would replace the baselines above:

- A compatibility model trained on outfit datasets such as Polyvore, instead of raw CLIP similarity plus rules.
- Garment segmentation, so background never leaks into the colours.
- Fine-tuned or few-shot category detection, using the corrections saved in this prototype.
- Richer personalisation: more ratings, colour-pair features, context such as weather or occasion, and a held-out test set of future ratings.
