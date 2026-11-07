"""Train, tune, evaluate and save five classifiers on the UCI Dry Bean dataset.

Run once on your machine / the BITS Virtual Lab:

    python train_models.py

What it produces
----------------
model/*.joblib            one fitted sklearn Pipeline per classifier (+ label encoder)
model/metadata.json       feature order, class names, best params, metrics, versions
test_data.csv             the held-out 20 % test split (features + target) for the app
results/*.csv|png         comparison tables, confusion matrices, EDA figures
README.md                 the auto-filled blocks (dataset stats, results, observations)

Options:  --data-dir data  --seed 2026  --test-size 0.2  --quick  --no-readme
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import time
import warnings
from pathlib import Path

# scikit-learn >= 1.9 warns that SVC(probability=True) is deprecated; it still works
# (and the app pins the training version), so keep the console output readable.
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from data_loader import TARGET_COL, load_clean  # noqa: E402
from evaluation import METRIC_ORDER, compute_metrics, confusion, confusion_figure, metrics_table  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RESULT_DIR = ROOT / "results"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


# --------------------------------------------------------------------------- #
# Model definitions + hyper-parameter grids (searched with 5-fold stratified CV)
# --------------------------------------------------------------------------- #
def build_candidates(seed: int, quick: bool) -> dict:
    scale = lambda: ("scale", StandardScaler())  # noqa: E731

    grids = {
        "Logistic Regression": {"clf__C": [0.1, 1, 10, 100]},
        "Decision Tree": {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [4, 6, 8, 12, None],
            "clf__min_samples_leaf": [1, 3, 5, 10],
        },
        "KNN": {
            "clf__n_neighbors": [3, 5, 7, 9, 13, 17],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
        "Naive Bayes": {"clf__var_smoothing": np.logspace(-11, -5, 7)},
        "Support Vector Machine": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]},
    }
    if quick:  # tiny grids, for a fast smoke test
        grids = {k: {p: list(v)[:2] for p, v in g.items()} for k, g in grids.items()}

    return {
        "Logistic Regression": (
            Pipeline([scale(), ("clf", LogisticRegression(max_iter=5000, random_state=seed))]),
            grids["Logistic Regression"],
        ),
        "Decision Tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=seed))]),
            grids["Decision Tree"],
        ),
        "KNN": (
            Pipeline([scale(), ("clf", KNeighborsClassifier())]),
            grids["KNN"],
        ),
        "Naive Bayes": (
            Pipeline([("clf", GaussianNB())]),
            grids["Naive Bayes"],
        ),
        "Support Vector Machine": (
            # probability is switched on after tuning (needed for AUC) - it slows
            # fitting ~5x, so the grid search itself runs without it.
            Pipeline([scale(), ("clf", SVC(kernel="rbf", random_state=seed))]),
            grids["Support Vector Machine"],
        ),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def save_eda_figures(df: pd.DataFrame, features: list[str]) -> int:
    """Class balance + correlation heat-map. Returns # of feature pairs with |r|>0.9."""
    order = df[TARGET_COL].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=order.index, y=order.values, hue=order.index, palette="viridis", legend=False, ax=ax)
    for i, v in enumerate(order.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title("Class distribution (after removing duplicates)")
    ax.set_xlabel("")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.3, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "feature_correlation.png", dpi=150)
    plt.close(fig)

    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    return int((corr.abs().where(upper) > 0.9).sum().sum())


def save_result_figures(table: pd.DataFrame, cms: dict, class_names: list[str]) -> None:
    for name, cm in cms.items():
        fig = confusion_figure(cm, class_names, normalize=False, title=f"{name} - confusion matrix (test set)")
        fig.savefig(RESULT_DIR / f"confusion_{MODEL_FILES[name].replace('.joblib', '')}.png", dpi=150)
        plt.close(fig)

    ax = table.plot(kind="bar", figsize=(10, 5), width=0.8, colormap="Set2", edgecolor="white")
    ax.set_ylim(max(table.min().min() - 0.1, -1.0), 1.0)
    ax.set_title("Test-set metrics by model")
    ax.set_xlabel("")
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "metrics_comparison.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# README auto-fill
# --------------------------------------------------------------------------- #
def replace_block(text: str, name: str, content: str) -> tuple[str, bool]:
    pattern = re.compile(rf"(<!-- {name}_START -->)(.*?)(<!-- {name}_END -->)", re.S)
    if not pattern.search(text):
        return text, False
    return pattern.sub(lambda m: f"{m.group(1)}\n{content.strip()}\n{m.group(3)}", text), True


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * len(df.columns)) + "|"]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([f"**{idx}**"] + cells) + " |")
    return "\n".join(lines)


def dataset_block(df, info, features, n_train, n_test, n_corr_pairs) -> str:
    counts = df[TARGET_COL].value_counts()
    rows = ["| Class | Samples | Share |", "|---|---:|---:|"]
    for cls, n in counts.items():
        rows.append(f"| {cls} | {n} | {n / len(df):.1%} |")
    return (
        f"- **Instances used:** {info['rows_clean']} (raw file: {info['rows_raw']}; "
        f"{info['duplicates_removed']} exact duplicate rows removed, {info['missing_cells']} missing cells)\n"
        f"- **Features:** {len(features)} numeric columns | **Target:** `{TARGET_COL}` ({counts.size} classes)\n"
        f"- **Split:** {n_train} train / {n_test} test (stratified 80/20)\n"
        f"- **Strongly correlated feature pairs (correlation above 0.9):** {n_corr_pairs}\n\n"
        + "\n".join(rows)
    )


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    return np.diag(cm) / np.clip(cm.sum(axis=1), 1, None)


def top_confusion(cm: np.ndarray, names: list[str]) -> tuple[str, str, int]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    return names[i], names[j], int(off[i, j])


def observations_block(table, summary, cms, names, params, n_corr_pairs) -> tuple[str, str]:
    """Data-driven first draft of the observations table + winner sentence."""
    rank = table["MCC"].rank(ascending=False, method="min").astype(int)
    best = rank.idxmin()
    rows = ["| ML Model Name | Observation about model performance |", "|---|---|"]

    for name in table.index:
        m, s, cm = table.loc[name], summary[name], cms[name]
        rec = per_class_recall(cm)
        weak = names[int(rec.argmin())]
        a, p, c = top_confusion(cm, names)
        gap = round((s["train_accuracy"] - m["Accuracy"]) * 100, 1) + 0.0  # + 0.0 avoids "-0.0"
        bp = params[name]
        if name == "Logistic Regression":
            why = ("It draws linear boundaries on standardised features, so it cannot bend around classes "
                   f"that overlap non-linearly (best C = {bp.get('clf__C')}).")
        elif name == "Decision Tree":
            why = ("Axis-aligned splits need no scaling and stay interpretable; depth and leaf size were tuned "
                   f"by CV (depth = {bp.get('clf__max_depth')}, min leaf = {bp.get('clf__min_samples_leaf')}) to limit overfitting.")
        elif name == "KNN":
            why = (f"It classifies by neighbourhood vote on standardised features (best k = {bp.get('clf__n_neighbors')}, "
                   f"weights = {bp.get('clf__weights')}); it keeps the whole training set, so prediction cost grows with data size.")
        elif name == "Naive Bayes":
            if n_corr_pairs >= 3:
                why = ("It assumes independent, Gaussian-distributed features, but "
                       f"{n_corr_pairs} feature pairs have a correlation above 0.9 (the size measures such as "
                       "Area, Perimeter and ConvexArea), which violates that assumption and typically costs accuracy.")
            else:
                why = "It assumes independent, Gaussian-distributed features; it is fast and needs almost no tuning."
        else:
            why = ("The RBF kernel learns smooth non-linear boundaries once features are standardised "
                   f"(best C = {bp.get('clf__C')}, gamma = {bp.get('clf__gamma')}).")
        rows.append(
            f"| **{name}** | Accuracy {m['Accuracy']:.2%}, macro-F1 {m['F1']:.3f}, MCC {m['MCC']:.3f} "
            f"(rank {rank[name]} of {len(table)} by MCC). {why} "
            + (f"Train accuracy is 100% by construction with distance weighting (each training point is its own "
               f"nearest neighbour), so the {gap:.1f}-point train-vs-test gap overstates overfitting; CV macro-F1 "
               f"({s['cv_f1_macro']:.3f}) closely matches test macro-F1 ({m['F1']:.3f}). "
               if name == "KNN" and bp.get("clf__weights") == "distance" else
               f"Train-vs-test accuracy gap is {gap:.1f} points ({'noticeable overfitting' if gap >= 5 else 'it generalises well'}). ")
            + f"Weakest class: {weak} (recall {rec.min():.2%}); most frequent mistake: {a} predicted as {p} ({c} cases). |"
        )

    runner = table["MCC"].drop(best).idxmax()
    winner = (
        f"**Winner: {best}.** It has the highest MCC ({table.loc[best, 'MCC']:.3f}), the metric that is most "
        f"robust to class imbalance, and also scores Accuracy {table.loc[best, 'Accuracy']:.2%}, "
        f"AUC {table.loc[best, 'AUC']:.4f} and macro-F1 {table.loc[best, 'F1']:.3f} on the held-out test set. "
        f"Runner-up: {runner} (MCC {table.loc[runner, 'MCC']:.3f})."
    )
    rows.append(f"| **Overall winner for this dataset** | {winner} |")
    return "\n".join(rows), winner


def update_readme(readme: Path, blocks: dict[str, str]) -> None:
    if not readme.exists():
        print("[readme] README.md not found - skipped")
        return
    text = readme.read_text(encoding="utf-8")
    for name, content in blocks.items():
        text, ok = replace_block(text, name, content)
        if not ok:
            print(f"[readme] marker {name} not found - skipped")
    readme.write_text(text, encoding="utf-8")
    print("[readme] auto-filled blocks in README.md")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--quick", action="store_true", help="tiny grids for a smoke test")
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    df, info = load_clean(ROOT / args.data_dir)
    features = [c for c in df.columns if c != TARGET_COL]
    assert len(features) >= 12 and len(df) >= 500, "dataset does not meet the assignment minimums"
    print(f"[data] {len(df)} rows, {len(features)} features, {df[TARGET_COL].nunique()} classes")

    n_corr_pairs = save_eda_figures(df, features)

    encoder = LabelEncoder().fit(df[TARGET_COL])
    class_names = list(encoder.classes_)
    X, y = df[features], encoder.transform(df[TARGET_COL])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # Test file used by the Streamlit app (features + readable target label)
    test_df = X_te.copy()
    test_df[TARGET_COL] = encoder.inverse_transform(y_te)
    test_df.reset_index(drop=True).to_csv(ROOT / "test_data.csv", index=False)
    print(f"[data] wrote test_data.csv ({len(test_df)} rows)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    n_classes = len(class_names)
    test_rows, test_rows_weighted, summary, cms, best_params, meta_models = {}, {}, {}, {}, {}, {}

    for name, (pipe, grid) in build_candidates(args.seed, args.quick).items():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_tr, y_tr)
        model = search.best_estimator_
        if name == "Support Vector Machine":
            model = clone(model).set_params(clf__probability=True).fit(X_tr, y_tr)
        fit_s = time.time() - t0

        pred, proba = model.predict(X_te), model.predict_proba(X_te)
        test_rows[name] = compute_metrics(y_te, pred, proba, n_classes, "macro")
        test_rows_weighted[name] = compute_metrics(y_te, pred, proba, n_classes, "weighted")
        cms[name] = confusion(y_te, pred, n_classes)
        best_params[name] = search.best_params_
        summary[name] = {
            "cv_f1_macro": float(search.best_score_),
            "train_accuracy": float((model.predict(X_tr) == y_tr).mean()),
            "tuning_and_fit_seconds": round(fit_s, 1),
        }
        joblib.dump(model, MODEL_DIR / MODEL_FILES[name], compress=3)
        print(f"[{name}] CV macro-F1 {search.best_score_:.4f} | test acc {test_rows[name]['Accuracy']:.4f} "
              f"| {fit_s:.0f}s | {search.best_params_}")
        meta_models[name] = {
            "file": MODEL_FILES[name],
            "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
            **summary[name],
        }

    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")

    table = metrics_table(test_rows)
    table.to_csv(RESULT_DIR / "metrics_comparison.csv", float_format="%.4f")
    metrics_table(test_rows_weighted).to_csv(RESULT_DIR / "metrics_comparison_weighted_avg.csv", float_format="%.4f")
    pd.DataFrame(summary).T.to_csv(RESULT_DIR / "model_training_summary.csv", float_format="%.4f")
    save_result_figures(table, cms, class_names)

    metadata = {
        "dataset": "UCI Dry Bean Dataset (id 602)",
        "target_column": TARGET_COL,
        "feature_names": features,
        "class_names": class_names,
        "random_state": args.seed,
        "test_size": args.test_size,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "models": meta_models,
        "test_metrics_macro": {k: {m: float(v) for m, v in r.items()} for k, r in test_rows.items()},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== Test-set comparison (macro-averaged precision/recall/F1) ===")
    print(table.round(4).to_string())

    if not args.no_readme:
        obs, _ = observations_block(table, summary, cms, class_names, best_params, n_corr_pairs)
        update_readme(ROOT / "README.md", {
            "DATASET_STATS": dataset_block(df, info, features, len(X_tr), len(X_te), n_corr_pairs),
            "RESULTS_TABLE": md_table(table),
            "OBSERVATIONS": obs,
        })

    print(f"\nDone. IMPORTANT: this machine used scikit-learn=={sklearn.__version__}. "
          f"Set the same pin in requirements.txt (scikit-learn=={sklearn.__version__}) so the "
          "deployed Streamlit app can unpickle the saved models.")


if __name__ == "__main__":
    main()
