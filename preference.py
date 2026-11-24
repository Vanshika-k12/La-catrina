"""Preference model: scikit-learn Logistic Regression trained on the user's Like / Dislike ratings.

Pipeline:  outfit -> named, interpretable features (planner.outfit_features)
                  -> StandardScaler -> LogisticRegression -> P(user likes outfit)

* Retrained from scratch on all saved ratings whenever they change (cheap: tens of examples).
* Not trained until there is enough feedback (MIN_TOTAL ratings with MIN_PER_CLASS likes AND
  dislikes). Until then `ready` is False and the planner falls back to the baseline score.
* Deterministic: lbfgs solver, no randomness.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MIN_TOTAL = 6        # ratings needed before the model is used at all
MIN_PER_CLASS = 2    # ... and at least this many likes and this many dislikes
CV_MIN_PER_CLASS = 3  # held-out evaluation needs at least this many of each class
C_REGULARISATION = 0.5  # smaller = stronger L2 penalty = more cautious probabilities (few ratings)


def make_model():
    # class_weight="balanced": a user who mostly clicks Like should not make the model
    # predict "like" for everything.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C_REGULARISATION, class_weight="balanced",
                                            max_iter=1000))


def readiness(y):
    """(ready, reason) for a list of 0/1 labels."""
    n_like, n_dislike = int(sum(y)), int(len(y) - sum(y))
    if len(y) < MIN_TOTAL or n_like < MIN_PER_CLASS or n_dislike < MIN_PER_CLASS:
        need = []
        if len(y) < MIN_TOTAL:
            need.append(f"{MIN_TOTAL - len(y)} more rating(s)")
        if n_like < MIN_PER_CLASS:
            need.append(f"{MIN_PER_CLASS - n_like} more like(s)")
        if n_dislike < MIN_PER_CLASS:
            need.append(f"{MIN_PER_CLASS - n_dislike} more dislike(s)")
        return False, "Needs " + ", ".join(need) + " before personalisation starts."
    return True, f"Trained on {len(y)} ratings ({n_like} likes, {n_dislike} dislikes)."


class PreferenceModel:
    def __init__(self, feature_names):
        self.feature_names = list(feature_names)
        self.pipe = None
        self.n_like = self.n_dislike = 0
        self.ready, self.reason = readiness([])

    # -- training -----------------------------------------------------------
    def vectorise(self, feature_dicts):
        return np.array([[d.get(n, 0.0) for n in self.feature_names] for d in feature_dicts],
                        dtype=float).reshape(len(feature_dicts), len(self.feature_names))

    def fit(self, feature_dicts, y):
        y = [int(v) for v in y]
        self.n_like, self.n_dislike = sum(y), len(y) - sum(y)
        self.ready, self.reason = readiness(y)
        self.pipe = None
        if self.ready:
            self.pipe = make_model().fit(self.vectorise(feature_dicts), y)
        return self

    @property
    def n_used(self):
        return self.n_like + self.n_dislike if self.ready else 0

    # -- prediction ---------------------------------------------------------
    def predict_proba(self, feature_dicts):
        """P(like) for each outfit; None when the model is not ready."""
        if not self.ready:
            return None
        return self.pipe.predict_proba(self.vectorise(feature_dicts))[:, 1]

    def contributions(self, feature_dict):
        """Per-feature contribution to the log-odds of 'like' for one outfit:
        coefficient x standardised feature value. Positive pushes towards 'like'."""
        scaler, lr = self.pipe[0], self.pipe[1]
        z = (self.vectorise([feature_dict])[0] - scaler.mean_) / scaler.scale_
        return dict(zip(self.feature_names, lr.coef_[0] * z))

    def coefficients(self):
        """Learned weight per feature (on standardised features, so sizes are comparable).
        Features that never varied in the ratings get weight 0 and are left out."""
        scaler, lr = self.pipe[0], self.pipe[1]
        return {n: float(c) for n, c, v in zip(self.feature_names, lr.coef_[0], scaler.var_)
                if v > 0}


def evaluate(feature_names, feature_dicts, y):
    """Honest evaluation on the user's ratings.

    * training metrics: model scored on the same ratings it learned from (optimistic).
    * cross-validated metrics: StratifiedKFold; the scaler and model are re-fitted inside every
      fold (one Pipeline), so no information from the held-out ratings leaks into training.
    * majority baseline: accuracy of always predicting the more common label.
    """
    y = np.array([int(v) for v in y])
    out = {"n": len(y), "n_like": int(y.sum()), "n_dislike": int(len(y) - y.sum())}
    ready, reason = readiness(list(y))
    out["ready"], out["reason"] = ready, reason
    if not ready:
        return out
    X = PreferenceModel(feature_names).vectorise(feature_dicts)

    def metrics(pred):
        return {"accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0)}

    out["train"] = metrics(make_model().fit(X, y).predict(X))
    out["majority_baseline_accuracy"] = max(y.mean(), 1 - y.mean())
    k = min(5, out["n_like"], out["n_dislike"])
    if min(out["n_like"], out["n_dislike"]) >= CV_MIN_PER_CLASS:
        folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        out["cv"] = metrics(cross_val_predict(make_model(), X, y, cv=folds))
        out["cv_folds"] = k
    else:
        out["cv"] = None
        out["cv_reason"] = (f"Held-out evaluation needs at least {CV_MIN_PER_CLASS} likes and "
                            f"{CV_MIN_PER_CLASS} dislikes.")
    return out
