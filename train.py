"""Train and evaluate the classifier. Splits by whole capture run so test traffic is never seen in training."""
import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from features import FEATURES, build_windows


def scores(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return dict(accuracy=accuracy_score(y, pred), precision=p, recall=r, f1=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/frames_train.csv.gz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    frames = pd.read_csv(a.data)
    w = build_windows(frames, a.window)
    X, y, groups = w[FEATURES], w["y"], w["run"]
    print(f"{len(w):,} windows | attack windows: {int(y.sum())} ({y.mean():.2%}) | runs: {groups.nunique()}")

    # Hold out ~30% of RUNS for testing
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=a.seed).split(X, y, groups))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced",
                                               random_state=a.seed, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=a.seed),
    }
    results = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        results[name] = scores(yte, m.predict(Xte))

    # Naive baseline: "more than 10 deauth frames/s means attack"
    results["Threshold baseline (deauth/s >= 10)"] = scores(yte, (Xte["deauth_rate"] >= 10).astype(int))
    # Majority-class baseline: shows why accuracy alone is misleading on imbalanced data
    results["Always 'normal'"] = scores(yte, np.zeros(len(yte), dtype=int))

    rf = models["RandomForest"]
    pred = rf.predict(Xte)
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    cv = cross_val_score(RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced",
                                                random_state=a.seed, n_jobs=-1),
                         X, y, groups=groups, cv=GroupKFold(5), scoring="f1")

    # Where does it struggle? Recall on low-rate attack windows vs high-rate ones
    low = (yte == 1) & (Xte["deauth_rate"] < 10)
    high = (yte == 1) & (Xte["deauth_rate"] >= 10)
    recall_low = float((pred[low.values] == 1).mean()) if low.any() else float("nan")
    recall_high = float((pred[high.values] == 1).mean()) if high.any() else float("nan")

    print("\nTest-set results (held-out runs)")
    print(pd.DataFrame(results).T.round(4).to_string())
    print("\nRandom Forest confusion matrix  [rows = true, cols = predicted]")
    print(pd.DataFrame(cm, index=["true normal", "true attack"], columns=["pred normal", "pred attack"]))
    print(f"\n5-fold run-grouped CV F1: {cv.mean():.3f} +/- {cv.std():.3f}")
    print(f"Recall on attack windows with <10 deauth/s: {recall_low:.3f} (n={int(low.sum())}) | "
          f">=10 deauth/s: {recall_high:.3f} (n={int(high.sum())})")

    # --- outputs ---
    joblib.dump(rf, f"{a.out}/wids_rf.joblib")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay(cm, display_labels=["normal", "attack"]).plot(ax=ax, cmap="Blues", colorbar=False,
                                                                       values_format="d")
    ax.set_title(f"Random Forest - accuracy {results['RandomForest']['accuracy']:.4f}")
    fig.tight_layout()
    fig.savefig(f"{a.out}/confusion_matrix.png", dpi=200)

    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    imp.plot.barh(ax=ax, color="#2a6f97")
    ax.set_xlabel("importance")
    ax.set_title("Which features matter")
    fig.tight_layout()
    fig.savefig(f"{a.out}/feature_importance.png", dpi=200)

    with open(f"{a.out}/metrics.json", "w") as f:
        json.dump(dict(results=results, confusion_matrix=cm.tolist(), cv_f1_mean=cv.mean(), cv_f1_std=cv.std(),
                       recall_low_rate=recall_low, recall_high_rate=recall_high,
                       feature_importance=imp.to_dict(), n_windows=len(w), n_attack_windows=int(y.sum())),
                  f, indent=2, default=float)
    print(f"\nsaved model, confusion_matrix.png, feature_importance.png, metrics.json -> {a.out}/")


if __name__ == "__main__":
    main()
