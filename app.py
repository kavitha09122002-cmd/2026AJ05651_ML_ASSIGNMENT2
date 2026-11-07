"""Dry Bean Classifier Lab - Streamlit front-end for the saved models.

The app NEVER retrains anything: it loads the pickled pipelines from ./model,
scores whatever test CSV you upload, and reports the evaluation metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import classification_report

from evaluation import METRIC_ORDER, compute_metrics, confusion, confusion_figure

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

st.set_page_config(page_title="Dry Bean Classifier Lab", page_icon="🫘", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      div[data-testid="stMetric"] {background:#f4f7f6; border:1px solid #dfe7e4;
          border-radius:10px; padding:10px 14px;}
      div[data-testid="stMetricLabel"] p {font-weight:600; color:#40504b;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Artifact loading (cached - models are read from disk once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading saved models...")
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    models = {name: joblib.load(MODEL_DIR / info["file"]) for name, info in meta["models"].items()}
    return meta, encoder, models


try:
    meta, encoder, models = load_artifacts()
except Exception as exc:  # missing files or scikit-learn version mismatch
    st.error(
        "Could not load the saved models from the `model/` folder.\n\n"
        f"`{type(exc).__name__}: {exc}`\n\n"
        "Run `python train_models.py` first, and make sure the scikit-learn version in "
        "`requirements.txt` matches the one used for training."
    )
    st.stop()

FEATURES: list[str] = meta["feature_names"]
TARGET: str = meta["target_column"]
CLASSES: list[str] = meta["class_names"]
N_CLASSES = len(CLASSES)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def prepare(df: pd.DataFrame):
    """Validate the uploaded frame; return (clean_df, y_true_encoded) or stop with a message."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        st.error(f"The CSV is missing required column(s): {', '.join(missing)}")
        st.caption("Required columns: " + ", ".join(FEATURES + [TARGET]))
        st.stop()

    df = df[FEATURES + [TARGET]]
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df[TARGET] = df[TARGET].astype(str).str.strip().str.title()
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < before:
        st.warning(f"Dropped {before - len(df)} row(s) with missing / non-numeric values.")

    unknown = sorted(set(df[TARGET]) - set(CLASSES))
    if unknown:
        st.warning(f"Ignoring rows with labels the models were not trained on: {', '.join(unknown)}")
        df = df[df[TARGET].isin(CLASSES)].reset_index(drop=True)
    if df.empty:
        st.error("No usable rows left after validation.")
        st.stop()
    return df, encoder.transform(df[TARGET])


def score(model_name: str, X: pd.DataFrame, y_true: np.ndarray, average: str):
    model = models[model_name]
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return pred, proba, compute_metrics(y_true, pred, proba, N_CLASSES, average)


def style_table(df: pd.DataFrame):
    try:
        return df.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8d5")
    except Exception:  # styler needs jinja2 - fall back to plain table
        return df.round(4)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Controls")
    model_name = st.selectbox("Classification model", list(models.keys()))
    average = st.radio(
        "Averaging for Precision / Recall / F1",
        ["macro", "weighted"],
        help="macro = every bean class counts equally; weighted = classes weighted by support.",
    )
    st.divider()
    upload = st.file_uploader("Upload test CSV", type="csv",
                              help="Must contain the 16 feature columns and the 'Class' column.")
    use_sample = st.checkbox("Use the bundled test_data.csv", value=upload is None and SAMPLE_CSV.exists(),
                             disabled=not SAMPLE_CSV.exists())
    if SAMPLE_CSV.exists():
        st.download_button("Download sample test CSV", SAMPLE_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
    st.divider()
    if meta.get("sklearn_version") != sklearn.__version__:
        st.warning(f"Models were trained with scikit-learn {meta['sklearn_version']}; "
                   f"this server runs {sklearn.__version__}.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🫘 Dry Bean Classifier Lab")
st.caption(
    "Predict the variety of a dry bean from 16 shape measurements - UCI Dry Bean dataset, "
    f"{N_CLASSES} classes, 5 pre-trained classifiers. Upload a test CSV to evaluate them."
)

if upload is not None:
    raw = pd.read_csv(upload)
    source = f"uploaded file `{upload.name}`"
elif use_sample and SAMPLE_CSV.exists():
    raw = pd.read_csv(SAMPLE_CSV)
    source = "bundled `test_data.csv`"
else:
    st.info("Upload a test CSV in the sidebar (or tick the bundled sample) to begin.")
    tab_about_only = st.tabs(["About the models"])[0]
    with tab_about_only:
        ref = pd.DataFrame(meta["test_metrics_macro"]).T[METRIC_ORDER]
        st.subheader("Reference results from training (held-out 20 % split)")
        st.dataframe(style_table(ref))
    st.stop()

data, y_true = prepare(raw)
X = data[FEATURES]
st.success(f"Evaluating on {source}: **{len(data):,} rows**, {len(FEATURES)} features, "
           f"{data[TARGET].nunique()} classes present.")

tab_eval, tab_compare, tab_pred, tab_about = st.tabs(
    ["📊 Evaluation", "🏁 Compare all models", "🔍 Predictions", "ℹ️ About"]
)

# ---- Evaluation of the selected model -------------------------------------- #
with tab_eval:
    pred, proba, m = score(model_name, X, y_true, average)
    st.subheader(model_name)
    cols = st.columns(6)
    for col, key in zip(cols, METRIC_ORDER):
        col.metric(key, "n/a" if np.isnan(m[key]) else f"{m[key]:.4f}")
    if np.isnan(m["AUC"]):
        st.caption("AUC needs at least two classes present in the uploaded file.")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("**Confusion matrix**")
        normalize = st.toggle("Show row-normalised (per-class recall)", value=False)
        cm = confusion(y_true, pred, N_CLASSES)
        fig = confusion_figure(cm, CLASSES, normalize=normalize)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        present = np.unique(np.concatenate([y_true, pred]))
        rep = classification_report(
            y_true, pred, labels=present, target_names=[CLASSES[i] for i in present],
            output_dict=True, zero_division=0,
        )
        rep_df = pd.DataFrame(rep).T
        rep_df["support"] = rep_df["support"].astype(int)
        st.dataframe(rep_df.round(4))
        worst = rep_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")["recall"].idxmin()
        st.caption(f"Hardest class for this model on this file: **{worst}** "
                   f"(recall {rep_df.loc[worst, 'recall']:.2%}).")

# ---- Side-by-side comparison ------------------------------------------------ #
with tab_compare:
    rows = {name: score(name, X, y_true, average)[2] for name in models}
    comp = pd.DataFrame(rows).T[METRIC_ORDER]
    comp.index.name = "ML Model"
    st.subheader("All five models on the same data")
    st.dataframe(style_table(comp))
    st.caption("Green = best value in each column.")

    best_model = comp["MCC"].idxmax()
    st.success(f"Best model on this file by MCC: **{best_model}** (MCC {comp.loc[best_model, 'MCC']:.4f}).")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    comp.plot(kind="bar", ax=ax, width=0.8, colormap="Set2", edgecolor="white")
    ax.set_ylim(max(float(np.nanmin(comp.values)) - 0.1, -1.0), 1.0)
    ax.set_xlabel("")
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.3), frameon=False)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.download_button("Download comparison as CSV", comp.round(6).to_csv().encode(),
                       file_name="model_comparison.csv", mime="text/csv")

# ---- Row-level predictions --------------------------------------------------- #
with tab_pred:
    pred, proba, _ = score(model_name, X, y_true, average)
    out = data.copy()
    out["Predicted"] = encoder.inverse_transform(pred)
    out["Confidence"] = proba.max(axis=1).round(4)
    out["Correct"] = out["Predicted"] == out[TARGET]
    only_wrong = st.checkbox("Show only misclassified rows", value=False)
    shown = out[~out["Correct"]] if only_wrong else out
    st.write(f"{len(shown):,} of {len(out):,} rows shown - "
             f"{(~out['Correct']).sum():,} misclassified by **{model_name}**.")
    st.dataframe(shown.head(500))
    st.download_button("Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv", mime="text/csv")

# ---- About ------------------------------------------------------------------ #
with tab_about:
    st.subheader("Dataset")
    st.write(
        f"**{meta['dataset']}** - {len(FEATURES)} geometric features extracted from images of "
        f"seven registered dry bean varieties: {', '.join(CLASSES)}."
    )
    st.write("Features: " + ", ".join(f"`{f}`" for f in FEATURES))
    st.subheader("Training setup")
    st.write(
        f"Stratified split: {meta['n_train']:,} train / {meta['n_test']:,} test "
        f"(random_state={meta['random_state']}). Each model is a scikit-learn Pipeline tuned with "
        "5-fold cross-validated grid search on macro-F1. Saved best parameters:"
    )
    params = pd.DataFrame(
        {n: {"CV macro-F1": round(i["cv_f1_macro"], 4), "Best params": str(i["best_params"])}
         for n, i in meta["models"].items()}
    ).T
    st.dataframe(params)
    st.subheader("Reference test results from training (macro averages)")
    st.dataframe(style_table(pd.DataFrame(meta["test_metrics_macro"]).T[METRIC_ORDER]))
