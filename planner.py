"""Weekly outfit planner.

Scoring (hybrid recommender):
  compatibility = W_COMPAT * mean CLIP image similarity + W_HARMONY * (mean HSV harmony + 1) / 2
  compat_norm   = compatibility min-max scaled to 0..1 over this wardrobe's possible outfits
                  (raw compatibility varies in a narrow band, so without rescaling even a small
                  preference weight would swamp it)
  final         = (1 - pref_weight) * compat_norm + pref_weight * P(like)      if the model is ready
  final         = compatibility                                                otherwise (fallback)
Guard: for outfits in the bottom COMPAT_FLOOR_QUANTILE of compatibility, P(like) is capped at 0.5,
so the user's likes can push a poor match down but never pull it up past good matches.
"""
import itertools
from collections import Counter

import numpy as np

from colours import COLOUR_NAMES, is_neutral, name_to_rgb, pair_harmony
from vision import CATEGORIES

TOPS = {"shirt", "T-shirt"}
BOTTOMS = {"jeans", "trousers", "skirt"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
KINDS = ["neutral", "analogous", "complementary", "clash", "mixed"]

W_COMPAT, W_HARMONY = 0.5, 0.5   # baseline compatibility score weights
PREF_WEIGHT = 0.3                # default: final = 0.7 * compatibility + 0.3 * P(like)
COMPAT_FLOOR_QUANTILE = 0.25     # bottom 25% compatibility: likes can't lift these
MAX_USES = 3                     # max days a single piece may appear
MAX_CORES = 300                  # cap on top+bottom+shoe combos expanded with layers
LAYER_OPTIONS = 3                # best N jackets / accessories considered per combo
MIN_EXPLAIN_LOGODDS = 0.3        # a feature must move the log-odds this much to be cited

# Named, interpretable outfit features used by the preference model.
FEATURE_NAMES = ([f"cat:{c}" for c in CATEGORIES]            # which garment types are in it
                 + [f"colour:{c}" for c in COLOUR_NAMES]     # which main colours are in it
                 + [f"harmony:{k}" for k in KINDS]           # share of colour pairs of each kind
                 + ["clip_similarity", "neutral_share", "n_colours"])

# Readable phrases for explanations: (phrase when high / present, phrase when low / absent)
_HARMONY_TEXT = {"neutral": "neutral colour pairings", "analogous": "similar-hue colours",
                 "complementary": "contrasting (complementary) colours",
                 "clash": "clashing bright colours", "mixed": "loosely matched colours"}


def feature_phrase(name, high=True):
    kind, _, val = name.partition(":")
    if kind == "cat":
        return {"accessory": "an accessory", "jacket": "a jacket layer",
                "footwear": "footwear", "jeans": "jeans", "trousers": "trousers"}.get(
            val, f"a {val}") if high else None
    if kind == "colour":
        return f"{val} pieces" if high else None
    if kind == "harmony":
        return _HARMONY_TEXT[val] if high else f"fewer {_HARMONY_TEXT[val]}"
    return {"clip_similarity": ("closely matching pieces", "less matchy pieces"),
            "neutral_share": ("mostly neutral outfits", "more colourful outfits"),
            "n_colours": ("outfits mixing several colours", "outfits with few colours")}[name][0 if high else 1]


def is_binary(name):
    return name.startswith(("cat:", "colour:"))


class OutfitScorer:
    def __init__(self, items, pref_model=None, pref_weight=PREF_WEIGHT):
        self.items = items
        self.index = {it["id"]: i for i, it in enumerate(items)}
        E = np.array([it["embedding"] for it in items], dtype=float)
        self.sim = E @ E.T
        n = len(items)
        self.harm = np.zeros((n, n))
        self.kind = [[""] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                a, b = items[i], items[j]
                s, k = pair_harmony(a["main_colour"], name_to_rgb(a["main_colour"]),
                                    b["main_colour"], name_to_rgb(b["main_colour"]))
                self.harm[i, j], self.kind[i][j] = s, k
        self.pref_model = pref_model
        self.pref_weight = pref_weight
        self._compat_stats = None

    @property
    def personalised(self):
        return self.pref_model is not None and self.pref_model.ready and self.pref_weight > 0

    def _stats_over_candidates(self):
        """(min, max, floor) of raw compatibility over all possible outfits in this wardrobe."""
        if self._compat_stats is None:
            bases = [self.base_score(s) for s in candidate_sets(self)] or [0.0]
            self._compat_stats = (min(bases), max(bases),
                                  float(np.quantile(bases, COMPAT_FLOOR_QUANTILE)))
        return self._compat_stats

    def compat_floor(self):
        """Compatibility below which likes cannot lift an outfit (a quantile of all candidates)."""
        return self._stats_over_candidates()[2]

    def compat_norm(self, base):
        lo, hi, _ = self._stats_over_candidates()
        return (base - lo) / (hi - lo) if hi > lo else 1.0

    def stats(self, idx):
        pairs = list(itertools.combinations(idx, 2))
        compat = float(np.mean([self.sim[i, j] for i, j in pairs]))
        harmony = float(np.mean([self.harm[i, j] for i, j in pairs]))
        kinds = Counter(self.kind[i][j] for i, j in pairs)
        return compat, harmony, kinds, len(pairs)

    def base_score(self, idx):
        compat, harmony, _, _ = self.stats(idx)
        return W_COMPAT * compat + W_HARMONY * (harmony + 1) / 2

    def features(self, idx):
        """Interpretable outfit features (dict name -> value) for the preference model."""
        compat, _, kinds, n_pairs = self.stats(idx)
        cats = {self.items[i]["category"] for i in idx}
        cols = [self.items[i]["main_colour"] for i in idx]
        f = {f"cat:{c}": 1.0 for c in cats}
        f.update({f"colour:{c}": 1.0 for c in set(cols)})
        f.update({f"harmony:{k}": kinds[k] / n_pairs for k in KINDS})
        f["clip_similarity"] = compat
        f["neutral_share"] = sum(is_neutral(c, name_to_rgb(c)) for c in cols) / len(cols)
        f["n_colours"] = float(len(set(cols)))
        return f

    def _combine(self, base, p):
        """final score from compatibility and P(like); returns (final, p_used, capped)."""
        if p is None:
            return base, None, False
        capped = base < self.compat_floor() and p > 0.5
        p_used = min(p, 0.5) if capped else p
        final = (1 - self.pref_weight) * self.compat_norm(base) + self.pref_weight * p_used
        return final, p_used, capped

    def score(self, idx):
        """{'score', 'base', 'pref', 'capped'}; pref is None when the model isn't used."""
        base = self.base_score(idx)
        p = (float(self.pref_model.predict_proba([self.features(idx)])[0])
             if self.personalised else None)
        final, _, capped = self._combine(base, p)
        return {"score": final, "base": base, "pref": p, "capped": capped}

    def personal_reason(self, idx, towards_like):
        """The feature that pushed this outfit's predicted preference most strongly in the
        given direction, as text. Grounded in the model: each feature's log-odds contribution
        is coefficient x standardised value."""
        contrib = self.pref_model.contributions(self.features(idx))
        sign = 1 if towards_like else -1
        name, value = max(contrib.items(), key=lambda kv: sign * kv[1])
        if sign * value < MIN_EXPLAIN_LOGODDS:
            return None
        x = self.features(idx).get(name, 0.0)
        above_avg = x > self.pref_model.pipe[0].mean_[self.pref_model.feature_names.index(name)]
        if is_binary(name):
            phrase = feature_phrase(name)
            if above_avg:   # the outfit has this piece / colour
                return (f"you tend to like {phrase}" if value > 0
                        else f"it has {phrase}, which you tend to dislike")
            return (f"it avoids {phrase}, which you tend to dislike" if value > 0
                    else f"it lacks {phrase}, which you tend to like")
        phrase = feature_phrase(name, high=above_avg)
        return (f"matches your preference for {phrase}" if value > 0
                else f"you tend to avoid {phrase}")

    def personal_line(self, cand):
        """One-line explanation of the personalised part of a candidate's ranking."""
        if not self.personalised:
            return None
        p, moved = cand["pref"], cand["base_rank"] - cand["rank"]
        move = (f"up {moved} place{'s' if moved != 1 else ''}" if moved > 0
                else f"down {-moved} place{'s' if moved != -1 else ''}" if moved < 0
                else "same place")
        line = f"Personalised: {p:.0%} predicted like ({move} vs. compatibility-only ranking)"
        reason = None if abs(p - 0.5) < 0.1 else self.personal_reason(cand["idx"], p > 0.5)
        if reason is None:
            line += "; your ratings don't strongly favour or penalise this outfit"
        else:
            line += f"; main factor: {reason}"
        if cand["capped"]:
            line += " (low compatibility, so likes can't lift it)"
        return line + "."

    def explain(self, idx):
        compat, _, kinds, _ = self.stats(idx)
        desc = " + ".join(f"{self.items[i]['main_colour']} {self.items[i]['category']}" for i in idx)
        accents = sorted({self.items[i]["main_colour"] for i in idx
                          if not is_neutral(self.items[i]["main_colour"],
                                            name_to_rgb(self.items[i]["main_colour"]))})
        if kinds["clash"]:
            colour = "some saturated colours clash"
        elif not accents:
            colour = "all-neutral palette, easy to wear"
        elif kinds["neutral"] == sum(kinds.values()):
            words = " and ".join(accents)
            colour = f"neutral base with {'an' if words[0] in 'aeiou' else 'a'} {words} accent"
        elif kinds["complementary"]:
            colour = "complementary colours add contrast"
        elif kinds["analogous"]:
            colour = "analogous colours look cohesive"
        else:
            colour = "mixed colours, moderately balanced"
        style = ("strong" if compat >= 0.75 else "good" if compat >= 0.6 else "loose")
        return f"{desc}: {colour}; {style} CLIP style match ({compat:.2f})."


def training_data(items, feedback):
    """Feature dicts and labels for every rating whose pieces are all still in the wardrobe."""
    base = OutfitScorer(items)
    X, y = [], []
    for f in feedback:
        if len(f["items"]) >= 2 and all(i in base.index for i in f["items"]):
            X.append(base.features(tuple(base.index[i] for i in f["items"])))
            y.append(int(f["label"]))
    return X, y


def build_scorer(items, feedback, pref_weight=PREF_WEIGHT):
    """Train the preference model on the saved ratings; return (scorer, model)."""
    from preference import PreferenceModel
    X, y = training_data(items, feedback)
    model = PreferenceModel(FEATURE_NAMES).fit(X, y)
    return OutfitScorer(items, model, pref_weight), model


def summarise_preferences(model, top=3):
    """(likes, dislikes): readable learned preferences, strongest first.
    Only features whose weight passes MIN_EXPLAIN_LOGODDS are reported."""
    if model is None or not model.ready:
        return [], []
    coefs = sorted(model.coefficients().items(), key=lambda kv: -abs(kv[1]))
    likes, dislikes = [], []
    for name, c in coefs:
        if abs(c) < MIN_EXPLAIN_LOGODDS:
            continue
        # for continuous features, a negative weight means a preference for the low end
        phrase = feature_phrase(name) if is_binary(name) else feature_phrase(name, high=c > 0)
        target = (likes if c > 0 else dislikes) if is_binary(name) else likes
        if phrase not in likes + dislikes:
            target.append(phrase)
    return likes[:top], dislikes[:top]


def group(items):
    g = {"top": [], "bottom": [], "dress": [], "footwear": [], "jacket": [], "accessory": []}
    for i, it in enumerate(items):
        c = it["category"]
        key = "top" if c in TOPS else "bottom" if c in BOTTOMS else c
        g[key].append(i)
    return g


def missing_parts(items):
    g = group(items)
    missing = []
    if not g["dress"] and not g["top"]:
        missing.append("a top (shirt / T-shirt) or a dress")
    if not g["dress"] and not g["bottom"]:
        missing.append("a bottom (jeans / trousers / skirt) or a dress")
    if not g["footwear"]:
        missing.append("footwear")
    return missing


def candidate_sets(scorer):
    """All outfits: (top+bottom | dress) + footwear, optionally + jacket and/or accessory."""
    g = group(scorer.items)
    cores = [(t, b) for t in g["top"] for b in g["bottom"]] + [(d,) for d in g["dress"]]
    core_shoes = [c + (f,) for c in cores for f in g["footwear"]]
    core_shoes.sort(key=lambda c: -scorer.base_score(c))
    core_shoes = core_shoes[:MAX_CORES]

    out = {}
    for cs in core_shoes:
        jackets = sorted(g["jacket"], key=lambda j: -scorer.base_score(cs + (j,)))[:LAYER_OPTIONS]
        accs = sorted(g["accessory"], key=lambda a: -scorer.base_score(cs + (a,)))[:LAYER_OPTIONS]
        for j in [None] + jackets:
            for a in [None] + accs:
                idx = cs + tuple(x for x in (j, a) if x is not None)
                out[frozenset(idx)] = idx
    return list(out.values())


def candidates(scorer):
    """All candidate outfits with final score, compatibility score, P(like) and two ranks:
    'rank' (by final score) and 'base_rank' (compatibility only, i.e. non-personalised)."""
    sets = candidate_sets(scorer)
    probs = (scorer.pref_model.predict_proba([scorer.features(s) for s in sets])
             if scorer.personalised and sets else [None] * len(sets))
    scored = []
    for idx, p in zip(sets, probs):
        base = scorer.base_score(idx)
        p = None if p is None else float(p)
        final, _, capped = scorer._combine(base, p)
        scored.append({"idx": idx, "score": final, "base": base, "pref": p, "capped": capped})
    for r, c in enumerate(sorted(scored, key=lambda c: (-c["base"], sorted(c["idx"]))), 1):
        c["base_rank"] = r
    scored.sort(key=lambda c: (-c["score"], sorted(c["idx"])))
    for r, c in enumerate(scored, 1):
        c["rank"] = r
    return scored


def rank_of(scorer, cands, idx):
    """Rank (1 = best) an outfit would have among the current candidates."""
    final = scorer.score(tuple(idx))["score"]
    return 1 + sum(1 for c in cands if c["score"] > final + 1e-12)


def _greedy(cands, n_days, max_uses, allow_repeat, disliked):
    usage, used_sets, week = Counter(), set(), []
    for _ in range(n_days):
        best, best_eff = None, -1e9
        for c in cands:
            key = frozenset(c["idx"])
            if not allow_repeat and (key in used_sets or key in disliked):
                continue
            if any(usage[i] >= max_uses for i in c["idx"]):
                continue
            # variety: prefer less-used pieces and outfits unlike earlier days
            reuse = np.mean([usage[i] for i in c["idx"]])
            overlap = max((len(key & s) / len(key | s) for s in used_sets), default=0.0)
            repeat = 1.0 if key in used_sets else 0.0
            eff = c["score"] - 0.12 * reuse - 0.2 * overlap - 1.0 * repeat
            if eff > best_eff:
                best, best_eff = c, eff
        if best is None:
            return None
        week.append(best)
        used_sets.add(frozenset(best["idx"]))
        usage.update(best["idx"])
    return week


def plan_week(scorer, n_days=7, disliked_ids=()):
    """Returns (week, notes, candidates). week is None if the wardrobe lacks required parts.

    disliked_ids: outfits (lists of item ids) rated thumbs-down; skipped unless the wardrobe is too small.
    """
    missing = missing_parts(scorer.items)
    if missing:
        return None, ["Can't build a plan yet - add " + ", ".join(missing) + "."], []
    cands = candidates(scorer)
    disliked = {frozenset(scorer.index[i] for i in ids) for ids in disliked_ids
                if all(i in scorer.index for i in ids)}
    notes = []
    week = _greedy(cands, n_days, MAX_USES, False, disliked)
    if week is None:
        notes.append(f"Wardrobe is small, so the 'max {MAX_USES} uses per piece' limit was relaxed.")
        week = _greedy(cands, n_days, 10 ** 9, False, disliked)
    if week is None:
        notes.append(f"Only {len(cands)} distinct outfits are possible, so some outfits "
                     "(possibly ones you rated 👎) repeat this week.")
        week = _greedy(cands, n_days, 10 ** 9, True, set())
    out = []
    for day, c in zip(DAYS, week):
        out.append({"day": day, "idx": list(c["idx"]),
                    "ids": [scorer.items[i]["id"] for i in c["idx"]],
                    "score": c["score"], "base": c["base"], "pref": c["pref"],
                    "rank": c["rank"], "base_rank": c["base_rank"],
                    "explanation": scorer.explain(c["idx"]),
                    "personal": scorer.personal_line(c)})
    return out, notes, cands
