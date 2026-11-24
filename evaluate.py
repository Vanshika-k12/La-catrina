"""Measure CLIP zero-shot category accuracy against your own labels,
or (with --preferences) evaluate the Like/Dislike preference model on your saved ratings.

Usage:  python evaluate.py [--labels labels.csv] [--photos data/wardrobe]
        python evaluate.py --preferences

labels.csv columns: filename,category,main_colour
Prints category accuracy, per-class precision/recall, a confusion matrix, and how often the
K-Means main colour name matches your main_colour label.
"""
import argparse
import os
import sys

import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

import vision
from colours import COLOUR_NAMES

CANON = {c.lower(): c for c in vision.CATEGORIES}
CANON.update({"tshirt": "T-shirt", "t shirt": "T-shirt", "tee": "T-shirt", "shoes": "footwear"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels.csv")
    ap.add_argument("--photos", default=os.path.join("data", "wardrobe"))
    ap.add_argument("--preferences", action="store_true",
                    help="evaluate the preference model on data/feedback.json instead")
    args = ap.parse_args()
    if args.preferences:
        return evaluate_preferences()

    if not os.path.exists(args.labels):
        sys.exit(f"{args.labels} not found. Create it with columns: filename,category,main_colour")
    try:
        df = pd.read_csv(args.labels, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        sys.exit(f"{args.labels} is empty. Add a header 'filename,category,main_colour' and one row per photo.")
    for col in ("filename", "category", "main_colour"):
        if col not in df.columns:
            sys.exit(f"{args.labels} is missing column '{col}'")

    vision.set_seeds()
    clip = vision.load_clip()
    y_true, y_pred, col_true, col_pred, skipped = [], [], [], [], []
    for _, row in df.iterrows():
        path = os.path.join(args.photos, row["filename"].strip())
        label = CANON.get(row["category"].strip().lower())
        if label is None:
            skipped.append(f"{row['filename']}: unknown category '{row['category']}'")
            continue
        if not os.path.exists(path):
            skipped.append(f"{row['filename']}: file not found in {args.photos}")
            continue
        img = Image.open(path)
        pred, _, _ = vision.analyse(clip, img)
        y_true.append(label)
        y_pred.append(pred)
        colour = row["main_colour"].strip().lower()
        pred_colour = vision.dominant_colours(img)[0]["name"]
        colour_note = ""
        if colour:
            if colour in COLOUR_NAMES:
                col_true.append(colour)
                col_pred.append(pred_colour)
                colour_note = f" | colour label={colour} predicted={pred_colour}" + \
                              ("" if colour == pred_colour else " (mismatch)")
            else:
                skipped.append(f"{row['filename']}: colour '{colour}' not in palette (colour not scored)")
        mark = "OK " if pred == label else "BAD"
        print(f"{mark} {row['filename']}: label={label} predicted={pred}{colour_note}")

    for s in skipped:
        print("SKIPPED", s)
    if not y_true:
        sys.exit("No usable rows - nothing to evaluate.")

    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    print(f"\nCategory accuracy: {correct}/{n} = {correct / n:.1%}")

    labels = [c for c in vision.CATEGORIES if c in set(y_true) | set(y_pred)]
    print("\nPer-class results:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true:{l}" for l in labels], columns=[f"pred:{l}" for l in labels])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("Confusion matrix (rows = your label, columns = CLIP prediction):")
        print(cm_df)

    if col_true:
        cc = sum(t == p for t, p in zip(col_true, col_pred))
        print(f"\nMain colour name match: {cc}/{len(col_true)} = {cc / len(col_true):.1%} "
              "(exact palette name; near-misses like navy/denim count as wrong)")


def evaluate_preferences():
    """Preference-model evaluation on the saved wardrobe + ratings (no CLIP needed)."""
    import planner
    import preference
    import storage
    from scipy.stats import spearmanr

    items, feedback = storage.load_wardrobe(), storage.load_feedback()
    X, y = planner.training_data(items, feedback)
    print(f"Ratings saved: {len(feedback)}; usable (all pieces still in wardrobe): {len(y)} "
          f"({sum(y)} likes, {len(y) - sum(y)} dislikes)")
    ev = preference.evaluate(planner.FEATURE_NAMES, X, y)
    if not ev["ready"]:
        print(ev["reason"])
        print("Nothing to evaluate yet - the app is using compatibility-only ranking.")
        return
    print(f"\nMajority-class baseline accuracy: {ev['majority_baseline_accuracy']:.2f}")
    print(f"{'metric':<10}{'training (optimistic)':>24}{'cross-validated':>18}")
    for m in ("accuracy", "precision", "recall", "f1"):
        cv = f"{ev['cv'][m]:.2f}" if ev["cv"] else "-"
        print(f"{m:<10}{ev['train'][m]:>24.2f}{cv:>18}")
    print(f"Cross-validation: {ev['cv_folds']}-fold stratified, scaler+model re-fitted per fold."
          if ev["cv"] else ev["cv_reason"])

    scorer, model = planner.build_scorer(items, feedback)
    likes, dislikes = planner.summarise_preferences(model)
    print(f"\nLearned preferences - likes: {likes or 'none clear'}; avoids: {dislikes or 'none clear'}")
    cands = planner.candidates(scorer)
    if len(cands) > 2:
        rho = spearmanr([c["base_rank"] for c in cands], [c["rank"] for c in cands]).statistic
        moved = sum(c["rank"] != c["base_rank"] for c in cands)
        print(f"Personalised vs compatibility-only ranking ({len(cands)} outfits): Spearman {rho:.2f}, "
              f"{moved} outfits changed rank (weight {planner.PREF_WEIGHT}).")
    print(f"\nNote: {len(y)} ratings is a very small dataset; treat these numbers as indicative only.")


if __name__ == "__main__":
    main()
