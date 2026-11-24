"""Tests for the personalisation pipeline (no CLIP model or real photos needed).

Run:  python test_personalisation.py        (or: python -m pytest test_personalisation.py)

Uses a synthetic wardrobe with seeded random "CLIP" embeddings and a temporary data folder
(LACATRINA_DATA_DIR), so your real data/ folder is never read or written.
"""
import json
import os
import tempfile
from collections import Counter

TMP = tempfile.mkdtemp(prefix="lacatrina_test_")
os.environ["LACATRINA_DATA_DIR"] = TMP  # must be set before storage is imported

import numpy as np  # noqa: E402

import planner  # noqa: E402
import preference  # noqa: E402
import storage  # noqa: E402

CAT_VEC = {}


def make_items():
    rng = np.random.default_rng(0)
    spec = [("t1", "T-shirt", "navy"), ("t2", "T-shirt", "red"), ("t3", "shirt", "white"),
            ("t4", "shirt", "navy"), ("t5", "T-shirt", "grey"),
            ("b1", "jeans", "denim"), ("b2", "trousers", "black"), ("b3", "skirt", "beige"),
            ("b4", "trousers", "red"),
            ("f1", "footwear", "white"), ("f2", "footwear", "brown"), ("f3", "footwear", "black"),
            ("j1", "jacket", "grey"), ("a1", "accessory", "tan"), ("d1", "dress", "red")]
    items = []
    for iid, cat, col in spec:
        base = CAT_VEC.setdefault(cat, rng.normal(size=512))
        e = base + 0.6 * rng.normal(size=512)
        items.append({"id": iid, "category": cat, "main_colour": col,
                      "embedding": (e / np.linalg.norm(e)).tolist(), "filename": iid + ".jpg"})
    return items


ITEMS = make_items()


def colours_of(idx):
    return {ITEMS[i]["main_colour"] for i in idx}


def rule_feedback(n_each):
    """Simulated user: likes outfits with navy, dislikes outfits with red (and no navy)."""
    cands = planner.candidates(planner.OutfitScorer(ITEMS))
    navy = [c["idx"] for c in cands if "navy" in colours_of(c["idx"])]
    red = [c["idx"] for c in cands if "red" in colours_of(c["idx"]) and "navy" not in colours_of(c["idx"])]
    fb = [{"items": [ITEMS[i]["id"] for i in idx], "label": 1, "time": k} for k, idx in enumerate(navy[:n_each])]
    fb += [{"items": [ITEMS[i]["id"] for i in idx], "label": 0, "time": 100 + k} for k, idx in enumerate(red[:n_each])]
    return fb, navy[n_each:], red[n_each:]


# ---------------------------------------------------------------- readiness / fallback
def test_not_ready_without_enough_feedback():
    for fb in ([], rule_feedback(2)[0][:3]):
        scorer, model = planner.build_scorer(ITEMS, fb)
        assert not model.ready and model.predict_proba([{}]) is None
        assert not scorer.personalised
        for c in planner.candidates(scorer):
            assert c["score"] == c["base"] and c["pref"] is None and c["rank"] == c["base_rank"]


def test_single_class_is_not_trained():
    fb, _, _ = rule_feedback(8)
    only_likes = [f for f in fb if f["label"] == 1]
    assert len(only_likes) >= preference.MIN_TOTAL
    _, model = planner.build_scorer(ITEMS, only_likes)
    assert not model.ready and "dislike" in model.reason


# ---------------------------------------------------------------- learning
def test_model_learns_and_generalises_to_unseen_outfits():
    fb, navy_unseen, red_unseen = rule_feedback(4)
    scorer, model = planner.build_scorer(ITEMS, fb)
    assert model.ready and model.n_used == 8
    p_navy = model.predict_proba([scorer.features(i) for i in navy_unseen])
    p_red = model.predict_proba([scorer.features(i) for i in red_unseen])
    assert p_navy.mean() > 0.5 > p_red.mean(), (p_navy.mean(), p_red.mean())
    likes, dislikes = planner.summarise_preferences(model, top=5)
    assert "navy pieces" in likes and "red pieces" in dislikes, (likes, dislikes)


def test_final_score_formula_and_compatibility_guard():
    fb, _, _ = rule_feedback(4)
    scorer, model = planner.build_scorer(ITEMS, fb, pref_weight=0.3)
    floor = scorer.compat_floor()
    seen_cap = False
    for c in planner.candidates(scorer):
        p_used = min(c["pref"], 0.5) if c["base"] < floor else c["pref"]
        assert abs(c["score"] - (0.7 * scorer.compat_norm(c["base"]) + 0.3 * p_used)) < 1e-9
        assert 0.0 <= scorer.compat_norm(c["base"]) <= 1.0
        seen_cap |= c["capped"]
        if c["capped"]:
            assert c["base"] < floor and c["pref"] > 0.5
    print(f"    guard applied to some candidates: {seen_cap}")


def test_weight_zero_falls_back_to_compatibility():
    fb, _, _ = rule_feedback(4)
    scorer, _ = planner.build_scorer(ITEMS, fb, pref_weight=0.0)
    assert not scorer.personalised
    assert all(c["rank"] == c["base_rank"] for c in planner.candidates(scorer))


def test_explanations_are_grounded():
    fb, _, _ = rule_feedback(4)
    scorer, model = planner.build_scorer(ITEMS, fb)
    week, _, _ = planner.plan_week(scorer, disliked_ids=[f["items"] for f in fb if not f["label"]])
    for d in week:
        assert d["personal"].startswith("Personalised:")
    # without a ready model there is no personal claim at all
    scorer0, _ = planner.build_scorer(ITEMS, fb[:2])
    week0, _, _ = planner.plan_week(scorer0)
    assert all(d["personal"] is None for d in week0)


# ---------------------------------------------------------------- planner
def check_week(week, disliked):
    assert len(week) == 7
    keys = [frozenset(d["ids"]) for d in week]
    assert len(set(keys)) == 7, "identical outfit repeated"
    assert not (set(keys) & disliked), "disliked outfit planned"
    uses = Counter(i for d in week for i in d["ids"])
    assert max(uses.values()) <= planner.MAX_USES, uses
    for d in week:
        cats = [next(it for it in ITEMS if it["id"] == i)["category"] for i in d["ids"]]
        has_core = "dress" in cats or (any(c in planner.TOPS for c in cats)
                                       and any(c in planner.BOTTOMS for c in cats))
        assert has_core and "footwear" in cats, cats


def test_planner_valid_with_and_without_personalisation():
    fb, _, _ = rule_feedback(4)
    disliked = {frozenset(f["items"]) for f in fb if f["label"] == 0}
    for feedback in ([], fb):
        scorer, _ = planner.build_scorer(ITEMS, feedback)
        week, notes, _ = planner.plan_week(scorer, disliked_ids=[f["items"] for f in feedback if not f["label"]])
        assert not notes, notes
        check_week(week, disliked if feedback else set())


def test_planner_small_wardrobe_relaxes_and_missing_parts():
    small = [it for it in ITEMS if it["id"] in ("t1", "b1", "f1")]
    week, notes, _ = planner.plan_week(planner.OutfitScorer(small))
    assert len(week) == 7 and len(notes) == 2
    _, notes, _ = planner.plan_week(planner.OutfitScorer([it for it in ITEMS if it["category"] != "footwear"]))
    assert notes and "footwear" in notes[0]


# ---------------------------------------------------------------- persistence
def test_feedback_persistence_no_duplicates_and_change():
    assert storage.FEEDBACK_FILE.startswith(TMP)
    storage.set_feedback(["b1", "t1", "f1"], 1)
    storage.set_feedback(["t1", "f1", "b1"], 1)          # same outfit, different order
    assert len(storage.load_feedback()) == 1
    storage.set_feedback(["f1", "b1", "t1"], 0)          # change of mind
    fb = storage.load_feedback()
    assert len(fb) == 1 and fb[0]["label"] == 0
    storage.set_feedback(["t2", "b2", "f2"], 1)
    storage.clear_feedback(["t1", "b1", "f1"])
    assert [f["items"] for f in storage.load_feedback()] == [["b2", "f2", "t2"]]


def test_legacy_duplicate_file_keeps_latest():
    legacy = [{"items": ["a", "b"], "label": 1, "time": 1.0},
              {"items": ["b", "a"], "label": 0, "time": 2.0},
              {"items": ["c", "d"], "label": 1, "time": 3.0}]
    with open(storage.FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(legacy, f)
    fb = storage.load_feedback()
    assert len(fb) == 2 and storage.feedback_lookup(fb)["a|b"] == 0


# ---------------------------------------------------------------- evaluation
def test_evaluation():
    fb, _, _ = rule_feedback(4)
    X, y = planner.training_data(ITEMS, fb)
    ev = preference.evaluate(planner.FEATURE_NAMES, X, y)
    assert ev["ready"] and ev["cv"] is not None and ev["cv_folds"] == 4
    for part in ("train", "cv"):
        assert all(0.0 <= v <= 1.0 for v in ev[part].values())
    print(f"    simulated user (8 ratings): train acc {ev['train']['accuracy']:.2f}, "
          f"CV acc {ev['cv']['accuracy']:.2f}, majority baseline {ev['majority_baseline_accuracy']:.2f}")
    fb2, _, _ = rule_feedback(2)
    X2, y2 = planner.training_data(ITEMS, fb2 + fb2[:0])
    assert not preference.evaluate(planner.FEATURE_NAMES, X2, y2)["ready"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # report every failure, not just the first
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed (temp data dir: {TMP})")
    raise SystemExit(1 if failed else 0)
