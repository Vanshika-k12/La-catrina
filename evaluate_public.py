"""Evaluate La Catrina's vision pipeline on a PUBLIC dataset (no training, evaluation only).

Dataset: "Fashion Product Images (Small)", Hugging Face mirror ashraq/fashion-product-images-small
(44,072 e-commerce product photos, 60x80 px, labelled with articleType and baseColour).
The mirror lists no licence; it is a copy of a dataset widely used for education/research.

Usage:
    python evaluate_public.py --download --per-class 100   # fetch a random sample (seed 0)
    python evaluate_public.py --run                        # evaluate; writes eval_results/

What is measured:
  1. Category accuracy of CLIP zero-shot (our 9 categories) + 95% Wilson confidence interval,
     per-class precision/recall/F1 and a confusion matrix.
  2. Prompt ablation on the same images: one generic prompt per class vs. our prompt ensemble
     before / after adding "tank top" + "camisole" to the T-shirt class.
  3. Main-colour accuracy of the K-Means pipeline: exact palette name and colour family.

Only dataset types/colours with an unambiguous match to our labels are used (see the maps below).
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

DATASET = "ashraq/fashion-product-images-small"
API = "https://datasets-server.huggingface.co"
OUT_DIR = os.path.join("eval_data", "fashion_product_sample")
RESULTS_DIR = "eval_results"
SEED = 0

# dataset articleType -> our category (only clear matches; everything else is excluded)
CATEGORY_MAP = {
    "shirt": ["Shirts"],
    "T-shirt": ["Tshirts", "Tops"],
    "jeans": ["Jeans"],
    "trousers": ["Trousers"],
    "skirt": ["Skirts"],
    "dress": ["Dresses"],
    "jacket": ["Jackets"],
    "footwear": ["Casual Shoes", "Sports Shoes", "Heels", "Formal Shoes", "Flats", "Sandals", "Flip Flops"],
    "accessory": ["Handbags", "Belts", "Caps", "Scarves", "Sunglasses", "Watches"],
}
# dataset baseColour -> our palette name (only exact equivalents; e.g. "Multi", "Silver" excluded)
COLOUR_MAP = {
    "Black": "black", "White": "white", "Blue": "blue", "Brown": "brown", "Grey": "grey",
    "Red": "red", "Green": "green", "Pink": "pink", "Navy Blue": "navy", "Purple": "purple",
    "Yellow": "yellow", "Beige": "beige", "Maroon": "maroon", "Orange": "orange", "Olive": "olive",
    "Cream": "cream", "Charcoal": "charcoal", "Lavender": "lavender", "Khaki": "khaki",
    "Teal": "teal", "Tan": "tan", "Mustard": "mustard",
}
# lenient metric: palette name -> colour family
FAMILY = {
    "black": "black", "charcoal": "grey", "grey": "grey", "light grey": "grey",
    "white": "white", "cream": "white", "beige": "beige", "khaki": "beige",
    "tan": "brown", "brown": "brown", "navy": "blue", "denim": "blue", "blue": "blue",
    "light blue": "blue", "teal": "green", "green": "green", "olive": "green",
    "yellow": "yellow", "mustard": "yellow", "orange": "orange", "red": "red",
    "maroon": "red", "pink": "pink", "purple": "purple", "lavender": "purple",
}
# prompt variants for the ablation (current = vision.PROMPTS)
PROMPTS_BEFORE_TANK = {"T-shirt": ["a photo of a t-shirt", "a photo of a plain tee"]}


# ---------------------------------------------------------------- download
def _get_json(url, tries=4):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lacatrina-eval/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def _filter_url(article_type, offset, length):
    where = urllib.parse.quote(f"\"articleType\"='{article_type}'")
    return (f"{API}/filter?dataset={DATASET}&config=default&split=train"
            f"&where={where}&offset={offset}&length={length}")


def download(per_class):
    """Random sample of `per_class` images per category, spread over its article types."""
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    jobs = []
    for cat, types in CATEGORY_MAP.items():
        counts = {t: _get_json(_filter_url(t, 0, 1))["num_rows_total"] for t in types}
        pool = [(t, i) for t in types for i in range(counts[t])]
        picks = rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)
        jobs += [(cat, *pool[p]) for p in sorted(picks)]
        print(f"{cat:10s} {sum(counts.values()):6d} available -> sampling {min(per_class, len(pool))}")

    def fetch(job):
        cat, atype, offset = job
        row = _get_json(_filter_url(atype, offset, 1))["rows"][0]["row"]
        path = os.path.join(OUT_DIR, f"{row['id']}.jpg")
        if not os.path.exists(path):
            req = urllib.request.Request(row["image"]["src"], headers={"User-Agent": "lacatrina-eval/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
                f.write(r.read())
        return {"id": row["id"], "filename": f"{row['id']}.jpg", "category": cat,
                "articleType": row["articleType"], "baseColour": row["baseColour"]}

    rows, failed = [], 0
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        for k, res in enumerate(pool.map(lambda j: _safe(fetch, j), jobs), 1):
            if res is None:
                failed += 1
            else:
                rows.append(res)
            if k % 100 == 0:
                print(f"  fetched {k}/{len(jobs)}")
    pd.DataFrame(rows).sort_values(["category", "id"]).to_csv(os.path.join(OUT_DIR, "metadata.csv"), index=False)
    size = sum(os.path.getsize(os.path.join(OUT_DIR, r["filename"])) for r in rows)
    print(f"Saved {len(rows)} images ({size / 1e6:.1f} MB) to {OUT_DIR}; {failed} failed.")


def _safe(fn, arg):
    try:
        return fn(arg)
    except Exception as e:
        print(f"  skip {arg}: {type(e).__name__}")
        return None


# ---------------------------------------------------------------- evaluation
def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def run():
    import torch
    from PIL import Image
    from sklearn.metrics import classification_report, confusion_matrix

    import vision

    meta_path = os.path.join(OUT_DIR, "metadata.csv")
    if not os.path.exists(meta_path):
        sys.exit("No sample found. Run: python evaluate_public.py --download --per-class 100")
    df = pd.read_csv(meta_path)
    vision.set_seeds()
    clip = vision.load_clip()

    # image embeddings + colours, computed once
    embs, colours = [], []
    t0 = time.time()
    for k, fn in enumerate(df["filename"], 1):
        img = Image.open(os.path.join(OUT_DIR, fn))
        _, _, e = vision.analyse(clip, img)
        embs.append(e)
        colours.append(vision.dominant_colours(img)[0]["name"])
        if k % 200 == 0:
            print(f"  processed {k}/{len(df)}")
    E = torch.tensor(embs, dtype=torch.float32)
    secs = (time.time() - t0) / len(df)

    def predict(prompts):
        T = vision.text_embeddings(clip, prompts)
        return [vision.CATEGORIES[i] for i in (100.0 * E @ T.T).argmax(dim=1).tolist()]

    variants = {
        "generic: one prompt 'a photo of a {category}'": {c: [f"a photo of a {c}"] for c in vision.CATEGORIES},
        "our ensemble, before tank/camisole": {**vision.PROMPTS, **PROMPTS_BEFORE_TANK},
        "our ensemble, current (used by the app)": vision.PROMPTS,
    }
    y = df["category"].tolist()
    preds = {name: predict(p) for name, p in variants.items()}
    current = preds["our ensemble, current (used by the app)"]

    lines = ["# Public-dataset evaluation (Fashion Product Images, small)", "",
             f"Sample: {len(df)} images, random (seed {SEED}), up to "
             f"{df['category'].value_counts().max()} per category. Images are 60x80 px product photos "
             "on white backgrounds. No training: CLIP is used zero-shot.", "",
             f"Average processing time: {secs:.2f} s per image on CPU (CLIP + colours).", "",
             "## 1. Category accuracy (CLIP zero-shot)", "",
             "| Prompt design | Accuracy | 95% CI |", "|---|---|---|"]
    for name, p in preds.items():
        k = sum(a == b for a, b in zip(y, p))
        lo, hi = wilson(k, len(y))
        lines.append(f"| {name} | {k}/{len(y)} = {k / len(y):.1%} | {lo:.1%} - {hi:.1%} |")
    labels = vision.CATEGORIES
    lines += ["", "### Per-class results (current prompts)", "", "```",
              classification_report(y, current, labels=labels, zero_division=0, digits=2), "```", "",
              "### Confusion matrix (rows = dataset label, columns = prediction)", "", "```"]
    cm = pd.DataFrame(confusion_matrix(y, current, labels=labels),
                      index=[f"true:{c}" for c in labels], columns=[f"pred:{c}" for c in labels])
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        lines += [cm.to_string(), "```", ""]

    # colour
    df["pred_colour"] = colours
    df["true_colour"] = df["baseColour"].map(COLOUR_MAP)
    col = df.dropna(subset=["true_colour"])
    exact = int((col["pred_colour"] == col["true_colour"]).sum())
    fam = int((col["pred_colour"].map(FAMILY) == col["true_colour"].map(FAMILY)).sum())
    n = len(col)
    lines += ["## 2. Main colour (K-Means + palette naming)", "",
              f"{n} of {len(df)} images have a dataset colour with an exact palette equivalent "
              f"({len(df) - n} excluded, e.g. Multi, Silver, Gold, Peach).", "",
              "| Metric | Result | 95% CI |", "|---|---|---|",
              f"| Exact palette name | {exact}/{n} = {exact / n:.1%} | "
              f"{wilson(exact, n)[0]:.1%} - {wilson(exact, n)[1]:.1%} |",
              f"| Same colour family | {fam}/{n} = {fam / n:.1%} | "
              f"{wilson(fam, n)[0]:.1%} - {wilson(fam, n)[1]:.1%} |", "",
              "Most common colour confusions (dataset -> predicted):", ""]
    wrong = col[col["pred_colour"] != col["true_colour"]]
    for (t, p), c in wrong.groupby(["true_colour", "pred_colour"]).size().sort_values(ascending=False).head(8).items():
        lines.append(f"- {t} -> {p}: {c}")
    lines += ["", "## Caveats", "",
              "- Product photos (clean white background, centred) differ from phone photos of clothes "
              "on a bed; colour is easier here, category may be harder at 60x80 px.",
              "- Dataset labels were mapped to our 9 categories; types with no clear match "
              "(kurtas, sarees, shorts, innerwear, cosmetics...) were excluded.",
              "- The dataset's colour label describes the product overall; multi-coloured items "
              "can legitimately have a different 'main' colour.", ""]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "public_eval.md"), "w", encoding="utf-8") as f:
        f.write(report)
    df.assign(pred_category=current).to_csv(os.path.join(RESULTS_DIR, "public_eval_predictions.csv"), index=False)
    print(report)
    print(f"\nSaved {RESULTS_DIR}/public_eval.md and public_eval_predictions.csv")


def mcnemar_p(a_correct, b_correct):
    """Exact McNemar test: are two classifiers' error rates different on the same images?"""
    from scipy.stats import binomtest
    a_only = sum(a and not b for a, b in zip(a_correct, b_correct))
    b_only = sum(b and not a for a, b in zip(a_correct, b_correct))
    if a_only + b_only == 0:
        return 1.0, a_only, b_only
    return binomtest(a_only, a_only + b_only, 0.5).pvalue, a_only, b_only


def calibrate(min_per_colour=3):
    """Learn colour-naming reference points on a DEVELOPMENT split, measure on a held-out TEST split.

    Reference point for a colour = mean CIELAB of the largest garment cluster over dev images
    labelled with that colour. Colours with fewer than `min_per_colour` dev images, or not in
    the dataset (e.g. denim, light grey), keep the ideal swatch.
    """
    from PIL import Image
    from sklearn.model_selection import train_test_split

    import colours
    import vision

    meta_path = os.path.join(OUT_DIR, "metadata.csv")
    if not os.path.exists(meta_path):
        sys.exit("No sample found. Run: python evaluate_public.py --download --per-class 100")
    df = pd.read_csv(meta_path)
    df["true_colour"] = df["baseColour"].map(COLOUR_MAP)
    df = df.dropna(subset=["true_colour"]).reset_index(drop=True)

    clusters = [vision.colour_clusters(Image.open(os.path.join(OUT_DIR, f))) for f in df["filename"]]
    df["main_lab"] = [colours.rgb_to_lab(c[int(np.argmax(n))]).tolist() for c, n in clusters]

    # stratified 50/50 split (colours seen only once go to the test half)
    counts = df["true_colour"].value_counts()
    strat = df[df["true_colour"].map(counts) >= 2]
    dev, test = train_test_split(strat, test_size=0.5, stratify=strat["true_colour"], random_state=SEED)
    test = pd.concat([test, df[df["true_colour"].map(counts) < 2]])

    calibrated, n_dev = {}, {}
    for colour, grp in dev.groupby("true_colour"):
        n_dev[colour] = len(grp)
        if len(grp) >= min_per_colour:
            calibrated[colour] = np.mean(np.array(grp["main_lab"].tolist()), axis=0).round(2).tolist()

    def predict(rows, reference):
        colours.set_naming_reference(reference)
        return [vision.name_clusters(*clusters[i])[0]["name"] for i in rows.index]

    results = {}
    for label, ref in (("ideal swatches (before)", None), ("calibrated (after)", calibrated)):
        pred = predict(test, ref)
        exact = [p == t for p, t in zip(pred, test["true_colour"])]
        fam = [FAMILY[p] == FAMILY[t] for p, t in zip(pred, test["true_colour"])]
        results[label] = {"pred": pred, "exact": exact, "family": fam}
    colours.set_naming_reference(colours._load_calibration())  # restore whatever is on disk

    before, after = results["ideal swatches (before)"], results["calibrated (after)"]
    p_exact, b_only_e, a_only_e = mcnemar_p(before["exact"], after["exact"])
    p_fam, _, _ = mcnemar_p(before["family"], after["family"])
    n = len(test)
    lines = ["# Colour-naming calibration (development / test split)", "",
             f"Images with an exact palette colour label: {len(df)}. Stratified 50/50 split "
             f"(seed {SEED}): **{len(dev)} development** images used to learn reference colours, "
             f"**{n} test** images never seen during calibration.", "",
             f"Calibrated {len(calibrated)} colours (>= {min_per_colour} development images each); "
             "the others keep the ideal swatch.", "",
             "## Held-out test results", "",
             "| Naming reference | Exact name | 95% CI | Same family | 95% CI |", "|---|---|---|---|---|"]
    for label, r in results.items():
        e, f = sum(r["exact"]), sum(r["family"])
        lines.append(f"| {label} | {e}/{n} = {e / n:.1%} | {wilson(e, n)[0]:.1%} - {wilson(e, n)[1]:.1%} "
                     f"| {f}/{n} = {f / n:.1%} | {wilson(f, n)[0]:.1%} - {wilson(f, n)[1]:.1%} |")
    lines += ["", f"Exact McNemar test (paired, same test images): exact-name p = {p_exact:.2g} "
                  f"({a_only_e} images fixed, {b_only_e} broken); family p = {p_fam:.2g}.", "",
              "## Per-colour exact accuracy on the test split (colours with >= 10 test images)", "",
              "| Colour | Test images | Before | After |", "|---|---|---|---|"]
    test = test.assign(before=before["exact"], after=after["exact"])
    for colour, grp in sorted(test.groupby("true_colour"), key=lambda kv: -len(kv[1])):
        if len(grp) >= 10:
            lines.append(f"| {colour} | {len(grp)} | {grp['before'].mean():.0%} | {grp['after'].mean():.0%} |")
    lines += ["", "## Learned reference colours (CIELAB)", "",
              "| Colour | Dev images | Ideal swatch L, a, b | Calibrated L, a, b |", "|---|---|---|---|"]
    for colour in sorted(calibrated, key=lambda c: -n_dev[c]):
        ideal = colours.rgb_to_lab(colours.PALETTE[colour]).round(1).tolist()
        lines.append(f"| {colour} | {n_dev[colour]} | {ideal} | {calibrated[colour]} |")
    lines += phone_photo_check(calibrated)
    colours.set_naming_reference(colours._load_calibration())  # restore whatever is on disk

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS_DIR, "colour_calibration.md"), "w", encoding="utf-8") as f:
        f.write(report)
    candidate = os.path.join(RESULTS_DIR, "colour_calibration_candidate.json")
    with open(candidate, "w", encoding="utf-8") as f:
        json.dump({"source": f"{DATASET} sample, development split (seed {SEED}, {len(dev)} images)",
                   "method": "mean CIELAB of largest garment cluster per labelled colour",
                   "n_dev": n_dev, "lab": calibrated}, f, indent=1)
    print(report)
    print(f"Saved {RESULTS_DIR}/colour_calibration.md and {candidate}. The app does NOT use it unless it is "
          "copied to colour_calibration.json in the project folder (check your own photos first: "
          "python evaluate.py).")


def phone_photo_check(calibrated, labels_csv="labels.csv", photos=os.path.join("data", "wardrobe")):
    """Domain-shift check: does the product-photo calibration help on the user's own phone photos?"""
    from PIL import Image

    import colours
    import vision

    if not os.path.exists(labels_csv):
        return ["", "## Transfer to phone photos", "", f"Skipped: {labels_csv} not found."]
    try:
        lab = pd.read_csv(labels_csv, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return ["", "## Transfer to phone photos", "", f"Skipped: {labels_csv} is empty."]
    rows = [(r["filename"].strip(), r["main_colour"].strip().lower()) for _, r in lab.iterrows()
            if r["main_colour"].strip().lower() in colours.PALETTE
            and os.path.exists(os.path.join(photos, r["filename"].strip()))]
    if not rows:
        return ["", "## Transfer to phone photos", "", "Skipped: no labelled photos with palette colours."]
    clusters = [vision.colour_clusters(Image.open(os.path.join(photos, f))) for f, _ in rows]
    preds = {}
    for label, ref in (("before", None), ("after", calibrated)):
        colours.set_naming_reference(ref)
        preds[label] = [vision.name_clusters(*c)[0]["name"] for c in clusters]
    n = len(rows)
    hits = {k: sum(p == t for p, (_, t) in zip(v, rows)) for k, v in preds.items()}
    out = ["", "## Transfer to the user's phone photos (domain-shift check)", "",
           f"The same calibration applied to the {n} labelled phone photos in `{labels_csv}` "
           "(clothes photographed at home, not product shots):", "",
           "| Naming reference | Exact name |", "|---|---|",
           f"| ideal swatches (before) | {hits['before']}/{n} |",
           f"| calibrated on product photos (after) | {hits['after']}/{n} |", "",
           "Predictions that changed:", ""]
    for (f, t), b, a in zip(rows, preds["before"], preds["after"]):
        if a != b:
            out.append(f"- {f} (label {t}): {b} -> {a}")
    out += ["", ("**Decision:** calibration helps phone photos too - consider installing it."
                 if hits["after"] > hits["before"] else
                 "**Decision:** the calibration does not transfer to phone photos (domain shift: "
                 "studio lighting vs room lighting), so the app keeps the ideal swatches. A calibration "
                 "for the app should be learned from labelled phone photos with its own dev / test split."),
            f"Only {n} phone photos, so this check is indicative."]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="learn colour-naming references on a dev split, report on a test split")
    args = ap.parse_args()
    if not (args.download or args.run or args.calibrate):
        ap.print_help()
    if args.download:
        download(args.per_class)
    if args.run:
        run()
    if args.calibrate:
        calibrate()
