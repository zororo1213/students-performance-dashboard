import warnings
warnings.filterwarnings("ignore")

# Use non-interactive backend (for Render, etc.)
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 9 models (same as your training script)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# XGBoost optional
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

plt.rcParams["font.family"] = "DejaVu Sans"

# -------------------------------------------------------------------
# GLOBAL STORAGE (for LDA prediction + model details)
# -------------------------------------------------------------------
trained = {
    "lda_model": None,
    "feature_cols": None,
    "income_categories": None,
    "income_map": None,
}

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def fig_to_pil(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail", "Success"])
    ax.set_yticklabels(["Fail", "Success"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.tight_layout()
    return fig_to_pil(fig)

# -------------------------------------------------------------------
# Preprocessing (aligned with your training script)
# -------------------------------------------------------------------
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

def study_hours_to_num(x):
    if pd.isna(x):
        return 0
    t = str(x).lower()
    if "less than 5" in t:
        return 3
    if "5 - 10" in t:
        return 7.5
    if "11 - 15" in t:
        return 13
    if "more than 15" in t:
        return 16
    try:
        return float(t)
    except Exception:
        return 0

def preprocess_df(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(missing)
            + "\nFound columns: "
            + ", ".join(df.columns)
        )

    # Target: Good/Excellent → 1, others → 0
    def map_performance(perf):
        if isinstance(perf, str) and ("good" in perf.lower() or "excellent" in perf.lower()):
            return 1
        return 0

    df["target"] = df["Performance"].apply(map_performance).astype(int)

    # Age
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Study hours
    df["Study_hours"] = df["Study hours per week"].apply(study_hours_to_num)

    # Family monthly income – normalize + encode
    income_raw = (
        df["Family monthly income"]
        .astype(str)
        .str.replace("\xa0", " ")
        .str.strip()
    )
    income_norm = income_raw.str.lower()
    unique_norm = sorted(income_norm.unique())
    income_norm_map = {cat: idx for idx, cat in enumerate(unique_norm)}
    df["Income_code"] = income_norm.map(income_norm_map)

    # For dropdown: use original display strings but map back to codes
    income_display = sorted(income_raw.unique())
    income_display_map = {}
    for disp in income_display:
        income_display_map[disp] = income_norm_map[disp.lower()]

    # Ordinal mappings (same as your training code)
    conf_map = {
        "very confident": 3,
        "somewhat confident": 2,
        "unsure": 1,
        "not confident": 0,
    }
    att_map = {
        "always (0 - 1 absence per month)": 3,
        "frequently (2 - 4 absences per month)": 2,
        "sometimes (5 - 7 absences per month)": 1,
        "rarely (more than 7 absences per month)": 0,
    }
    pun_map = {
        "always on time": 3,
        "occasionally late": 2,
        "frequently late": 1,
        "rarely on time": 0,
    }
    eng_map = {
        "very engaged": 3,
        "moderately engaged": 2,
        "slightly engaged": 1,
        "not engaged": 0,
    }
    str_map = {
        "always": 3,
        "frequently": 2,
        "sometimes": 1,
        "rarely": 0,
    }

    df["Confidence_code"] = df["Confidence"].astype(str).str.lower().map(conf_map).fillna(1)
    df["Attendance_code"] = df["Frequency of attendance"].astype(str).str.lower().map(att_map).fillna(1)
    df["Punctuality_code"] = df["Punctuality"].astype(str).str.lower().map(pun_map).fillna(1)
    df["Engagement_code"] = df["Class engagement"].astype(str).str.lower().map(eng_map).fillna(1)
    df["Stress_code"] = df["Frequency of stress"].astype(str).str.lower().map(str_map).fillna(1)

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
    y = df["target"]

    return X, y, feature_cols, income_display, income_display_map

# -------------------------------------------------------------------
# Build the 9 models
# -------------------------------------------------------------------
def build_models():
    models = [
        (LinearDiscriminantAnalysis(), "LDA"),
        (AdaBoostClassifier(random_state=42), "AdaBoost"),
        (
            xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            ),
            "XGBoost",
        ) if HAS_XGB else (None, "XGBoost"),
        (RandomForestClassifier(random_state=42, n_estimators=300), "RandomForest"),
        (LogisticRegression(max_iter=1000, random_state=42), "LogisticRegression"),
        (GaussianNB(), "NaiveBayes"),
        (DecisionTreeClassifier(random_state=42), "DecisionTree"),
        (SVC(probability=True, random_state=42), "SVM"),
        (KNeighborsClassifier(), "KNN"),
    ]
    if not HAS_XGB:
        models = [m for m in models if m[1] != "XGBoost"]
    return models

# -------------------------------------------------------------------
# Tab 1: Train & Compare Models
# -------------------------------------------------------------------
def train_and_compare(file_obj):
    if file_obj is None:
        return (
            "Please upload the survey CSV.",
            None,
            None,
            "No LDA report yet.",
            gr.update(),
            gr.update(),
            {},
        )

    # Load CSV
    try:
        df = pd.read_csv(file_obj.name)
    except Exception:
        file_obj.seek(0)
        df = pd.read_csv(file_obj)

    X, y, feature_cols, income_cats, income_map = preprocess_df(df)

    # 60/20/20 split (stratified, same random_state as thesis)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    models = build_models()
    rows = []
    model_details = {}

    lda_cm_test_img = None
    lda_report_text = None
    lda_model = None

    for model, name in models:
        if model is None:
            rows.append([name, np.nan, np.nan])
            continue

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        rows.append([name, val_acc, test_acc])

        cm_val = confusion_matrix(y_val, y_val_pred)
        cm_test = confusion_matrix(y_test, y_test_pred)
        cm_val_img = plot_confusion_matrix(cm_val, f"Validation Confusion Matrix — {name}")
        cm_test_img = plot_confusion_matrix(cm_test, f"Test Confusion Matrix — {name}")
        report_test = classification_report(y_test, y_test_pred, zero_division=0)

        model_details[name] = {
            "cm_val": cm_val_img,
            "cm_test": cm_test_img,
            "report_test": report_test,
        }

        if name == "LDA":
            lda_cm_test_img = cm_test_img
            lda_report_text = report_test
            lda_model = model

    summary_df = pd.DataFrame(rows, columns=["Model", "Validation Accuracy", "Test Accuracy"])
    summary_df = summary_df.sort_values("Test Accuracy", ascending=False)

    # Line graph of test accuracy (all models)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(summary_df["Model"], summary_df["Test Accuracy"], marker="o")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Test Accuracy of Models (Line Graph)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    line_img = fig_to_pil(fig)

    # Markdown summary + table
    summary_md = (
        f"### Dataset Split\n"
        f"- Train size: **{len(X_train)}**\n"
        f"- Validation size: **{len(X_val)}**\n"
        f"- Test size: **{len(X_test)}**\n\n"
        "### Model Accuracy Table\n"
        + summary_df.to_markdown(index=False)
    )

    # LDA classification report (Test)
    if lda_report_text is not None:
        lda_report_md = "```text\n" + lda_report_text + "\n```"
    else:
        lda_report_md = "LDA failed to train."

    # Save LDA + feature info for Tab 2 prediction
    trained["lda_model"] = lda_model
    trained["feature_cols"] = feature_cols
    trained["income_categories"] = income_cats
    trained["income_map"] = income_map

    # Dropdown update for income (Tab 2)
    income_dd_update = gr.update(
        choices=income_cats,
        value=income_cats[0] if income_cats else None,
    )

    # Dropdown update for model selection (per-model details in Tab 1)
    model_names = summary_df["Model"].tolist()
    model_dd_update = gr.update(
        choices=model_names,
        value=model_names[0] if model_names else None,
    )

    return (
        summary_md,
        line_img,
        lda_cm_test_img,
        lda_report_md,
        income_dd_update,
        model_dd_update,
        model_details,
    )

# -------------------------------------------------------------------
# Per-model detail renderer (Tab 1 dropdown)
# -------------------------------------------------------------------
def show_model_details(model_name, details_store):
    if not model_name or not details_store or model_name not in details_store:
        return None, None, "Select a model above to view its details."
    info = details_store[model_name]
    report_md = "```text\n" + info["report_test"] + "\n```"
    return info["cm_val"], info["cm_test"], report_md

# -------------------------------------------------------------------
# Tab 2: LDA Prediction System
# -------------------------------------------------------------------
CONF_OPTIONS = ["Very confident", "Somewhat confident", "Unsure", "Not confident"]
ATTEND_OPTIONS = [
    "Always (0 - 1 absence per month)",
    "Frequently (2 - 4 absences per month)",
    "Sometimes (5 - 7 absences per month)",
    "Rarely (more than 7 absences per month)",
]
PUNCT_OPTIONS = [
    "Always on time",
    "Occasionally late",
    "Frequently late",
    "Rarely on time",
]
ENGAGE_OPTIONS = [
    "Very engaged",
    "Moderately engaged",
    "Slightly engaged",
    "Not engaged",
]
STRESS_OPTIONS = ["Always", "Frequently", "Sometimes", "Rarely"]

def predict_single_student(
    age,
    study_hours_choice,
    income_cat,
    confidence,
    attendance,
    punctuality,
    engagement,
    stress,
):
    lda = trained["lda_model"]
    feature_cols = trained["feature_cols"]
    income_map = trained["income_map"]

    if lda is None or feature_cols is None or income_map is None:
        return "⚠️ Please upload the CSV and run training in Tab 1 first."

    if income_cat not in income_map:
        return "⚠️ Invalid family income selection."

    # Age
    try:
        age_val = float(age)
    except Exception:
        return "❌ Age must be numeric."

    # Study hours
    study_val = study_hours_to_num(study_hours_choice)

    # Codes (same mapping as preprocess_df)
    conf_map = {
        "very confident": 3,
        "somewhat confident": 2,
        "unsure": 1,
        "not confident": 0,
    }
    att_map = {
        "always (0 - 1 absence per month)": 3,
        "frequently (2 - 4 absences per month)": 2,
        "sometimes (5 - 7 absences per month)": 1,
        "rarely (more than 7 absences per month)": 0,
    }
    pun_map = {
        "always on time": 3,
        "occasionally late": 2,
        "frequently late": 1,
        "rarely on time": 0,
    }
    eng_map = {
        "very engaged": 3,
        "moderately engaged": 2,
        "slightly engaged": 1,
        "not engaged": 0,
    }
    str_map = {
        "always": 3,
        "frequently": 2,
        "sometimes": 1,
        "rarely": 0,
    }

    row = pd.DataFrame(
        [{
            "Age": age_val,
            "Study_hours": study_val,
            "Income_code": income_map[income_cat],
            "Confidence_code": conf_map[confidence.lower()],
            "Attendance_code": att_map[attendance.lower()],
            "Punctuality_code": pun_map[punctuality.lower()],
            "Engagement_code": eng_map[engagement.lower()],
            "Stress_code": str_map[stress.lower()],
        }]
    )[feature_cols]

    proba = lda.predict_proba(row)[0]
    prob_fail = float(proba[0])
    prob_success = float(proba[1])
    pred = 1 if prob_success >= 0.5 else 0

    if pred == 1:
        main = "✅ **Predicted: Success (Good/Excellent)**"
    else:
        main = "⚠️ **Predicted: At Risk / Below Good**"

    if prob_success >= 0.6:
        risk = "**Risk Level:** Likely Success"
    elif prob_success <= 0.4:
        risk = "**Risk Level:** Likely At Risk"
    else:
        risk = "**Risk Level:** Borderline / Uncertain"

    details = (
        "\n\n### Probability Breakdown\n"
        f"- Success (Good/Excellent): **{prob_success:.2%}**\n"
        f"- At Risk / Below Good: **{prob_fail:.2%}**"
    )

    return main + "\n\n" + risk + details

# -------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------
with gr.Blocks(title="Thesis Model Dashboard") as demo:
    gr.Markdown("## 🎓 Thesis Model Dashboard — Predicting Student Academic Performance")

    # State for per-model details
    model_details_state = gr.State({})

    # TAB 1: Train & Compare
    with gr.Tab("1️⃣ Train & Compare Models"):
        file_in = gr.File(label="Upload survey CSV (.csv)", file_types=[".csv"])
        run_btn = gr.Button("Run Training & Evaluation", variant="primary")

        summary_md = gr.Markdown()
        line_img = gr.Image(label="Test Accuracy of Models (Line Graph)")
        lda_cm_img = gr.Image(label="LDA Test Confusion Matrix")
        lda_report_md = gr.Markdown()

        gr.Markdown("### 🔍 Per-Model Detailed Results")
        model_select = gr.Dropdown(label="Select model", choices=[], value=None)

        with gr.Tabs():
            with gr.Tab("Validation Confusion Matrix"):
                cm_val_out = gr.Image()
            with gr.Tab("Test Confusion Matrix"):
                cm_test_out = gr.Image()
            with gr.Tab("Test Classification Report"):
                rep_out = gr.Markdown()

    # TAB 2: LDA Prediction System
    with gr.Tab("2️⃣ Predict Single Student (LDA)"):
        gr.Markdown("Use the trained **LDA** model (best performer) to predict a single student's outcome.")

        with gr.Row():
            age_in = gr.Number(label="Age", value=18)
            study_in = gr.Dropdown(
                label="Study hours per week",
                choices=[
                    "Less than 5 hours",
                    "5 - 10 hours",
                    "11 - 15 hours",
                    "More than 15 hours",
                ],
                value="5 - 10 hours",
            )

        income_in = gr.Dropdown(label="Family monthly income", choices=[], value=None)

        with gr.Row():
            conf_in = gr.Dropdown(label="Confidence", choices=CONF_OPTIONS, value="Somewhat confident")
            att_in = gr.Dropdown(label="Attendance", choices=ATTEND_OPTIONS, value=ATTEND_OPTIONS[0])

        with gr.Row():
            pun_in = gr.Dropdown(label="Punctuality", choices=PUNCT_OPTIONS, value=PUNCT_OPTIONS[0])
            eng_in = gr.Dropdown(label="Engagement", choices=ENGAGE_OPTIONS, value=ENGAGE_OPTIONS[1])

        stress_in = gr.Dropdown(label="Stress", choices=STRESS_OPTIONS, value="Sometimes")

        pred_btn = gr.Button("Predict", variant="primary")
        pred_out = gr.Markdown()

    # Callbacks
    run_btn.click(
        train_and_compare,
        inputs=[file_in],
        outputs=[
            summary_md,
            line_img,
            lda_cm_img,
            lda_report_md,
            income_in,
            model_select,
            model_details_state,
        ],
    )

    model_select.change(
        show_model_details,
        inputs=[model_select, model_details_state],
        outputs=[cm_val_out, cm_test_out, rep_out],
    )

    pred_btn.click(
        predict_single_student,
        inputs=[
            age_in,
            study_in,
            income_in,
            conf_in,
            att_in,
            pun_in,
            eng_in,
            stress_in,
        ],
        outputs=[pred_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
