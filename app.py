import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

import os
import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# Helpers (plots -> PIL images)
# -----------------------------
def fig_to_pil(fig, dpi=150, tight=True):
    buf = BytesIO()
    if tight:
        fig.tight_layout()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail", "Success"])
    ax.set_yticklabels(["Fail", "Success"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    # numbers
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center", color="black", fontsize=12)
    return fig_to_pil(fig)


def plot_bar(names, values, title, ylabel="Accuracy"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(names, rotation=45, ha="right")
    return fig_to_pil(fig)


def plot_feature_importance_or_coef(model, feat_names, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        ax.bar(feat_names, vals)
        ax.set_title(title)
        ax.set_ylabel("Importance")
    elif hasattr(model, "coef_"):
        coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        ax.bar(feat_names, coefs)
        ax.set_title(title)
        ax.set_ylabel("Coefficient")
    else:
        plt.close(fig)
        return None
    ax.set_xticklabels(feat_names, rotation=45, ha="right")
    fig.tight_layout()
    return fig_to_pil(fig)


# -----------------------------
# Data preprocessing (matches your notebook)
# -----------------------------
def preprocess_df(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    # target
    def map_performance(perf):
        if isinstance(perf, str) and ("Good" in perf or "Excellent" in perf):
            return 1
        return 0

    if "Performance" not in df.columns:
        raise ValueError("Column 'Performance' not found in CSV.")

    df["target"] = df["Performance"].apply(map_performance)

    # Age
    if "Age" not in df.columns:
        raise ValueError("Column 'Age' not found in CSV.")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Study hours
    if "Study hours per week" not in df.columns:
        raise ValueError("Column 'Study hours per week' not found in CSV.")

    def study_hours_to_num(x):
        if pd.isna(x):
            return 0
        x = str(x).lower()
        if "less than 5" in x:
            return 3
        elif "5 - 10" in x:
            return 7.5
        elif "11 - 15" in x:
            return 13
        elif "more than 15" in x:
            return 16
        else:
            try:
                return float(x)
            except Exception:
                return 0

    df["Study_hours"] = df["Study hours per week"].apply(study_hours_to_num)

    # Income (label encode)
    if "Family monthly income" not in df.columns:
        raise ValueError("Column 'Family monthly income' not found in CSV.")
    df["Family monthly income"] = (
        df["Family monthly income"].astype(str).str.replace("\xa0", " ").str.strip().str.lower()
    )
    income_le = LabelEncoder()
    df["Income_code"] = income_le.fit_transform(df["Family monthly income"])

    # Confidence
    if "Confidence" not in df.columns:
        raise ValueError("Column 'Confidence' not found in CSV.")
    confidence_map = {"very confident": 3, "somewhat confident": 2, "unsure": 1, "not confident": 0}
    df["Confidence"] = df["Confidence"].astype(str).str.lower()
    df["Confidence_code"] = df["Confidence"].map(confidence_map).fillna(1)

    # Attendance
    if "Frequency of attendance" not in df.columns:
        raise ValueError("Column 'Frequency of attendance' not found in CSV.")
    attendance_map = {
        "always (0 - 1 absence per month)": 3,
        "frequently (2 - 4 absences per month)": 2,
        "sometimes (5 - 7 absences per month)": 1,
        "rarely (more than 7 absences per month)": 0,
    }
    df["Frequency of attendance"] = df["Frequency of attendance"].astype(str).str.lower()
    df["Attendance_code"] = df["Frequency of attendance"].map(attendance_map).fillna(1)

    # Punctuality
    if "Punctuality" not in df.columns:
        raise ValueError("Column 'Punctuality' not found in CSV.")
    punctuality_map = {
        "always on time": 3,
        "occasionally late": 2,
        "frequently late": 1,
        "rarely on time": 0,
    }
    df["Punctuality"] = df["Punctuality"].astype(str).str.lower()
    df["Punctuality_code"] = df["Punctuality"].map(punctuality_map).fillna(1)

    # Engagement
    if "Class engagement" not in df.columns:
        raise ValueError("Column 'Class engagement' not found in CSV.")
    engagement_map = {
        "very engaged": 3,
        "moderately engaged": 2,
        "slightly engaged": 1,
        "not engaged": 0,
    }
    df["Class engagement"] = df["Class engagement"].astype(str).str.lower()
    df["Engagement_code"] = df["Class engagement"].map(engagement_map).fillna(1)

    # Stress
    if "Frequency of stress" not in df.columns:
        raise ValueError("Column 'Frequency of stress' not found in CSV.")
    stress_map = {"always": 3, "frequently": 2, "sometimes": 1, "rarely": 0}
    df["Frequency of stress"] = df["Frequency of stress"].astype(str).str.lower()
    df["Stress_code"] = df["Frequency of stress"].map(stress_map).fillna(1)

    feature_cols = [
        "Age",
        "Study_hours",
        "Income_code",
        "Confidence_code",
        "Attendance_code",
        "Punctuality_code",
        "Engagement_code",
        "Stress_code",
    ]

    X = df[feature_cols]
    y = df["target"].astype(int)

    return X, y, feature_cols


# ---------------------------------------
# Build chosen sklearn/xgb models
# ---------------------------------------
def build_models(use_class_weight=False):
    # class_weight only for certain models
    cw = "balanced" if use_class_weight else None

    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
        ),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=300),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42, class_weight=cw),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight=cw),
        "SVM": SVC(probability=True, random_state=42, class_weight=cw),
        "KNN": KNeighborsClassifier(),
    }
    return models


# ---------------------------------------
# Core run: train, evaluate, collect plots
# ---------------------------------------
def run_pipeline(file_obj, selected_models, use_class_weight):
    if file_obj is None:
        raise gr.Error("Please upload a CSV file first.")

    # Read CSV to df
    try:
        df = pd.read_csv(file_obj.name)
    except Exception:
        # gradio File objects sometimes need .read() buffer
        file_obj.seek(0)
        df = pd.read_csv(file_obj)

    # preprocess
    X, y, feat_names = preprocess_df(df)

    # split 60/20/20 with fixed seed like your code
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # build models
    all_models = build_models(use_class_weight)
    if not selected_models:
        # default: use all
        selected_models = list(all_models.keys())

    # collect results
    summary_rows = []
    panels = {}  # stored in State: per-model images/metrics for detail tab
    test_bars = []
    test_bar_names = []

    for name in selected_models:
        model = all_models[name]

        # fit once (not epoch loop; dashboards should be fast)
        model.fit(X_train, y_train)

        # validation metrics
        yv = model.predict(X_val)
        val_acc = accuracy_score(y_val, yv)

        # test metrics
        yt = model.predict(X_test)
        test_acc = accuracy_score(y_test, yt)

        summary_rows.append([name, val_acc, test_acc])
        test_bars.append(test_acc)
        test_bar_names.append(name)

        # confusion matrices
        cm_val = confusion_matrix(y_val, yv)
        cm_test = confusion_matrix(y_test, yt)
        cm_val_img = plot_confusion_matrix(cm_val, f"Validation Confusion Matrix — {name}")
        cm_test_img = plot_confusion_matrix(cm_test, f"Test Confusion Matrix — {name}")

        # ROC/PR if proba available
        roc_img = None
        pr_img = None
        if hasattr(model, "predict_proba"):
            try:
                yv_prob = model.predict_proba(X_val)[:, 1]
                fig1, ax1 = plt.subplots(figsize=(4.5, 3.8))
                RocCurveDisplay.from_predictions(y_val, yv_prob, ax=ax1, name=name)
                ax1.set_title(f"Validation ROC — {name}")
                roc_img = fig_to_pil(fig1)

                fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
                PrecisionRecallDisplay.from_predictions(y_val, yv_prob, ax=ax2, name=name)
                ax2.set_title(f"Validation Precision–Recall — {name}")
                pr_img = fig_to_pil(fig2)
            except Exception:
                pass

        # feature importance/coefficients
        fi_img = plot_feature_importance_or_coef(model, feat_names, f"Features — {name}")

        # store panel
        panels[name] = {
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "cm_val_img": cm_val_img,
            "cm_test_img": cm_test_img,
            "roc_img": roc_img,
            "pr_img": pr_img,
            "feat_img": fi_img,
        }

    # summary markdown
    df_sum = pd.DataFrame(summary_rows, columns=["Model", "Validation Accuracy", "Test Accuracy"])
    df_sum = df_sum.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    md_lines = ["### Summary Accuracy Table", "", df_sum.to_markdown(index=False)]

    # overall bar chart
    bar_img = plot_bar(df_sum["Model"].tolist(), df_sum["Test Accuracy"].tolist(),
                       "Test Accuracy of Models")

    return "\n".join(md_lines), bar_img, gr.update(choices=df_sum["Model"].tolist(),
                                                   value=df_sum["Model"].iloc[0]), panels


# ---------------------------------------
# Gradio UI
# ---------------------------------------
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# 🎓 Thesis Model Dashboard — Reproduce & Show Your Testing")
    gr.Markdown("Uploads your CSV and runs the exact pipeline (60/20/20 split, random_state=42).")

    with gr.Row():
        file_in = gr.File(label="Upload dataset CSV", file_types=[".csv"])
        use_cw = gr.Checkbox(label="Use class_weight='balanced' where supported (helps when one class dominates)",
                             value=False)

    all_model_names = [
        "LDA", "AdaBoost", "XGBoost", "RandomForest", "LogisticRegression",
        "NaiveBayes", "DecisionTree", "SVM", "KNN"
    ]
    model_checks = gr.CheckboxGroup(all_model_names, value=["LDA", "RandomForest",
                                                            "LogisticRegression", "NaiveBayes"],
                                    label="Models to run")

    run_btn = gr.Button("Run", variant="primary")

    summary_md = gr.Markdown()
    overall_img = gr.Image(label="Overall Results", height=360)
    # for the details section
    model_choice = gr.Dropdown(choices=[], label="Per-Model Details", interactive=True)
    panel_state = gr.State({})  # holds images/metrics per model

    with gr.Row():
        # Validation & Test confusion matrices
        cm_val_out = gr.Image(label="Validation Confusion Matrix")
        cm_test_out = gr.Image(label="Test Confusion Matrix")

    with gr.Row():
        roc_out = gr.Image(label="Validation ROC (if supported)")
        pr_out = gr.Image(label="Validation Precision-Recall (if supported)")

    feat_out = gr.Image(label="Feature Importance / Coefficients (if available)")

    # wire up
    def _run(file, selected, use_class_weight):
        # returns summary_md, bar_img, model_choice_update, panels (state)
        return run_pipeline(file, selected, use_class_weight)

    run_btn.click(
        _run,
        inputs=[file_in, model_checks, use_cw],
        outputs=[summary_md, overall_img, model_choice, panel_state],
        show_progress="full"
    )

    # when selecting a model, pull its images from state
    def _details(name, panels):
        if not name or not panels or name not in panels:
            return None, None, None, None, None
        p = panels[name]
        return (
            f"**{name}** — Val Acc: {p['val_acc']:.3f} | Test Acc: {p['test_acc']:.3f}",
            p["cm_val_img"],
            p["cm_test_img"],
            p["roc_img"],
            p["pr_img"],
            p["feat_img"]
        )

    header_md = gr.Markdown()
    model_choice.change(
        _details,
        inputs=[model_choice, panel_state],
        outputs=[header_md, cm_val_out, cm_test_out, roc_out, pr_out, feat_out]
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        inbrowser=False,
        debug=False
    )
