import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image  # for returning numpy arrays to Gradio
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
import traceback, warnings, os

# =========================
# Utilities (plots -> numpy)
# =========================
def _fig_to_numpy(fig):
    """Convert a Matplotlib figure to a NumPy array (so gr.Image(type='numpy') works)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf))

def _cm_numpy(cm, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail", "Success"]); ax.set_yticklabels(["Fail", "Success"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    return _fig_to_numpy(fig)

def _bar_numpy(names, scores, title="Test Accuracy of Selected Models"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, scores)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Accuracy"); ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _fig_to_numpy(fig)

def _feat_numpy(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        ylab = "Importance"; ttl = f"Feature Importance — {model_name}"
    elif hasattr(model, "coef_"):
        vals = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        ylab = "Coefficient"; ttl = f"Feature Coefficients — {model_name}"
    else:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(feature_names, vals)
    ax.set_title(ttl); ax.set_ylabel(ylab); ax.set_xlabel("Features")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return _fig_to_numpy(fig)

def _roc_numpy(y_true, y_score, title):
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
        ax.set_title(title)
        return _fig_to_numpy(fig)
    except Exception:
        return None

def _pr_numpy(y_true, y_score, title):
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
        ax.set_title(title)
        return _fig_to_numpy(fig)
    except Exception:
        return None

# =========================
# Data loading & preprocess
# =========================
EXPECTED_COLS = [
    "Age",
    "Study hours per week",
    "Family monthly income",
    "Confidence",
    "Frequency of attendance",
    "Punctuality",
    "Class engagement",
    "Frequency of stress",
    "Performance",
]

def _load_csv(file_obj):
    df = pd.read_csv(file_obj.name)
    df.columns = df.columns.str.strip()
    return df

def _preprocess(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    # Ensure required columns exist
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + "\nFound columns: "
            + ", ".join(df.columns)
        )

    def map_performance(perf):
        if isinstance(perf, str) and ("Good" in perf or "Excellent" in perf):
            return 1
        return 0

    df["target"] = df["Performance"].apply(map_performance)

    # pandas-3.0-safe assignment
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df["Age"].fillna(df["Age"].median())

    def study_hours_to_num(x):
        if pd.isna(x):
            return 0
        xx = str(x).lower()
        if "less than 5" in xx: return 3
        if "5 - 10" in xx: return 7.5
        if "11 - 15" in xx: return 13
        if "more than 15" in xx: return 16
        try: return float(xx)
        except: return 0

    df["Study_hours"] = df["Study hours per week"].apply(study_hours_to_num)

    df["Family monthly income"] = (
        df["Family monthly income"].astype(str).replace("\xa0", " ", regex=False)
        .str.strip().str.lower()
    )
    income_le = LabelEncoder()
    df["Income_code"] = income_le.fit_transform(df["Family monthly income"])

    confidence_map = {"very confident": 3, "somewhat confident": 2, "unsure": 1, "not confident": 0}
    df["Confidence"] = df["Confidence"].astype(str).str.lower()
    df["Confidence_code"] = df["Confidence"].map(confidence_map).fillna(1)

    attendance_map = {
        "always (0 - 1 absence per month)": 3,
        "frequently (2 - 4 absences per month)": 2,
        "sometimes (5 - 7 absences per month)": 1,
        "rarely (more than 7 absences per month)": 0,
    }
    df["Frequency of attendance"] = df["Frequency of attendance"].astype(str).str.lower()
    df["Attendance_code"] = df["Frequency of attendance"].map(attendance_map).fillna(1)

    punctuality_map = {
        "always on time": 3,
        "occasionally late": 2,
        "frequently late": 1,
        "rarely on time": 0,
    }
    df["Punctuality"] = df["Punctuality"].astype(str).str.lower()
    df["Punctuality_code"] = df["Punctuality"].map(punctuality_map).fillna(1)

    engagement_map = {"very engaged": 3, "moderately engaged": 2, "slightly engaged": 1, "not engaged": 0}
    df["Class engagement"] = df["Class engagement"].astype(str).str.lower()
    df["Engagement_code"] = df["Class engagement"].map(engagement_map).fillna(1)

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

# =========================
# Models
# =========================
ALL_MODEL_NAMES = [
    "LDA", "AdaBoost", "XGBoost", "RandomForest", "LogisticRegression",
    "NaiveBayes", "DecisionTree", "SVM", "KNN"
]

def _build_models(use_balanced=False):
    cw = "balanced" if use_balanced else None
    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric="logloss", random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, class_weight=cw),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight=cw),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight=cw),
        "SVM": SVC(probability=True, random_state=42, class_weight=cw),
        "KNN": KNeighborsClassifier(),
    }
    return models

# =========================
# Core pipeline (single fit per model)
# =========================
def run_pipeline(csv_file, models_to_run, use_balanced):
    logs = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if csv_file is None:
                return (
                    "Please upload a CSV.", None, None, "(no logs)", [],
                    "Select a model above.", "Select a model above.", None, None, None, None
                ), {}

            df = _load_csv(csv_file)
            X, y, feature_cols = _preprocess(df)

            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.4, stratify=y, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
            )

            models = _build_models(use_balanced)
            chosen = [m for m in models_to_run if m in models]
            if not chosen:
                return (
                    "Select at least one model.", None, None, "(no logs)", [],
                    "Select a model above.", "Select a model above.", None, None, None, None
                ), {}

            names, test_scores, rows = [], [], []
            panels = {}

            for name in chosen:
                try:
                    model = models[name]
                    model.fit(X_train, y_train)

                    yv = model.predict(X_val)
                    yt = model.predict(X_test)
                    val_acc = accuracy_score(y_val, yv)
                    test_acc = accuracy_score(y_test, yt)

                    cm_val = confusion_matrix(y_val, yv)
                    cm_test = confusion_matrix(y_test, yt)
                    rep_val = classification_report(y_val, yv, zero_division=0)
                    rep_test = classification_report(y_test, yt, zero_division=0)

                    yv_score = yt_score = None
                    if hasattr(model, "predict_proba"):
                        yv_score = model.predict_proba(X_val)[:, 1]
                        yt_score = model.predict_proba(X_test)[:, 1]
                    elif hasattr(model, "decision_function"):
                        yv_score = model.decision_function(X_val)
                        yt_score = model.decision_function(X_test)

                    img_cm_val = _cm_numpy(cm_val, f"Validation Confusion Matrix — {name}")
                    img_cm_test = _cm_numpy(cm_test, f"Test Confusion Matrix — {name}")
                    img_roc = _roc_numpy(y_val, yv_score, f"ROC — {name} (Validation)") if yv_score is not None else None
                    img_pr  = _pr_numpy(y_val, yv_score,  f"PR — {name} (Validation)")  if yv_score is not None else None
                    img_feat = _feat_numpy(model, feature_cols, name)

                    rows.append([name, round(val_acc, 4), round(test_acc, 4)])
                    names.append(name); test_scores.append(test_acc)

                    panels[name] = {
                        "val_report": rep_val,
                        "test_report": rep_test,
                        "cm_val": img_cm_val,
                        "cm_test": img_cm_test,
                        "roc": img_roc,
                        "pr": img_pr,
                        "feat": img_feat
                    }
                except Exception as me:
                    tb = traceback.format_exc(limit=1)
                    logs.append(f"❌ {name} failed: {me}\n{tb}")

            if not rows:
                head = "All selected models failed. See errors below."
                return (head, None, None, "\n".join(logs) or "(no logs)", [], "", "", None, None, None, None), panels

            summary_df = pd.DataFrame(rows, columns=["Model", "Validation Accuracy", "Test Accuracy"])
            bar_img = _bar_numpy(names, test_scores, "Test Accuracy of Selected Models")
            head = (
                f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} | "
                f"Stratified 60/20/20 | class_weight={'balanced' if use_balanced else 'None'}"
            )
            return (
                head, summary_df, bar_img, "\n".join(logs) or "(no errors)",
                sorted(panels.keys()),
                "Select a model above.", "Select a model above.", None, None, None, None
            ), panels

    except Exception as e:
        tb = traceback.format_exc()
        head = f"❌ Pipeline error: {e}"
        return (head, None, None, tb, [], "", "", None, None, None, None), {}

# =========================
# Gradio UI
# =========================
with gr.Blocks(title="Thesis Model Dashboard") as demo:
    gr.Markdown("## 🎓 Thesis Model Dashboard — Reproduce & Show Your Testing\nUploads your CSV and runs the exact pipeline (60/20/20, random_state=42).")

    with gr.Row():
        csv_in = gr.File(label="Upload dataset CSV", file_types=[".csv"])
        use_bal = gr.Checkbox(label="Use class_weight='balanced' where supported", value=False)

    pick = gr.CheckboxGroup(choices=ALL_MODEL_NAMES,
                            value=["LDA","RandomForest","LogisticRegression","NaiveBayes"],
                            label="Models to run")

    run_btn = gr.Button("Run", variant="primary")

    head_out   = gr.Markdown()
    summary_df = gr.Dataframe(label="Summary (Validation/Test Accuracies)")
    bar_img    = gr.Image(label="Test Accuracy Bar Chart", type="numpy")
    logs_md    = gr.Markdown(label="Errors / Logs")

    gr.Markdown("---")
    gr.Markdown("### 🔎 Per-Model Details")

    with gr.Row():
        model_sel = gr.Dropdown(choices=[], label="Select a model to view details", value=None)

    val_rep = gr.Markdown(label="Validation Report")
    test_rep = gr.Markdown(label="Test Report")
    with gr.Row():
        cm_val_img = gr.Image(label="Validation Confusion Matrix", type="numpy")
        cm_test_img = gr.Image(label="Test Confusion Matrix", type="numpy")
    with gr.Row():
        roc_img = gr.Image(label="Validation ROC (if supported)", type="numpy")
        pr_img  = gr.Image(label="Validation Precision–Recall (if supported)", type="numpy")
    feat_img = gr.Image(label="Feature Importance / Coefficients (if available)", type="numpy")

    panel_state = gr.State({})

    # 1) Run pipeline
    def _run(csv_file, selected_models, use_balanced):
        (head, df, bar, logs, model_list,
         val_txt, test_txt, cmv, cmt, roc, pr), panels = run_pipeline(csv_file, selected_models, use_balanced)

        return (
            head, df, bar, logs,
            gr.update(choices=model_list, value=(model_list[0] if model_list else None)),
            panel_state.update(panels)
        )

    run_btn.click(
        _run,
        inputs=[csv_in, pick, use_bal],
        outputs=[head_out, summary_df, bar_img, logs_md, model_sel, panel_state]
    )

    # 2) Render chosen model
    def _render_model(model_name, panels):
        if not model_name or model_name not in panels:
            msg = "Select a model above."
            return msg, msg, None, None, None, None, None
        p = panels[model_name]
        return (
            "```\n" + p["val_report"] + "\n```",
            "```\n" + p["test_report"] + "\n```",
            p["cm_val"], p["cm_test"],
            p["roc"], p["pr"], p["feat"]
        )

    model_sel.change(
        _render_model,
        inputs=[model_sel, panel_state],
        outputs=[val_rep, test_rep, cm_val_img, cm_test_img, roc_img, pr_img, feat_img]
    )

# ---- render.com launch block ----
if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        inbrowser=False,
        debug=False
    )
