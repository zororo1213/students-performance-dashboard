import warnings
warnings.filterwarnings("ignore")

# Safe backend + font so plots work on Render
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

from io import BytesIO
from PIL import Image

import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 9 models as in your thesis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# XGBoost is optional but we’ll include it (9th model)
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# -------------------------------------------------------------------
# GLOBALS – store LDA + feature columns + income categories/mapping
# -------------------------------------------------------------------
trained_artifacts = {
    "lda_model": None,
    "feature_cols": None,
    "income_categories": None,  # list of display strings for dropdown
    "income_map": None,         # dict: display_string -> integer_code
}


# -----------------------------
# Plot helpers
# -----------------------------
def fig_to_pil(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail", "Success"])
    ax.set_yticklabels(["Fail", "Success"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig_to_pil(fig)


# -----------------------------
# Preprocessing (aligned to thesis)
# -----------------------------
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
    """Convert survey text ranges to numeric values (same mapping as thesis)."""
    if pd.isna(x):
        return 0
    xx = str(x).lower()
    if "less than 5" in xx:
        return 3
    if "5 - 10" in xx:
        return 7.5
    if "11 - 15" in xx:
        return 13
    if "more than 15" in xx:
        return 16
    try:
        return float(xx)
    except Exception:
        return 0


def preprocess_df(df_raw):
    """
    Clean and encode the survey data:

    - Performance: Good/Excellent -> 1, others -> 0
    - Age: numeric, median imputation
    - Study hours: text → numeric
    - Family monthly income: create stable integer codes + display strings
    - Confidence, attendance, punctuality, engagement, stress: ordinal codes
    """
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

    # Target: Good/Excellent = 1 (Success), others = 0 (Below Good)
    def map_performance(perf):
        if isinstance(perf, str) and ("good" in perf.lower() or "excellent" in perf.lower()):
            return 1
        return 0

    df["target"] = df["Performance"].apply(map_performance).astype(int)

    # Age
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Study hours per week -> numeric
    df["Study_hours"] = df["Study hours per week"].apply(study_hours_to_num)

    # ----- Family monthly income -----
    # Keep original display text + normalized lower-case version
    income_raw = (
        df["Family monthly income"]
        .astype(str)
        .str.replace("\xa0", " ")
        .str.strip()
    )
    income_norm = income_raw.str.lower()

    # Create stable integer codes from normalized values
    unique_norm = sorted(income_norm.unique())
    income_map_norm = {cat: idx for idx, cat in enumerate(unique_norm)}
    df["Income_code"] = income_norm.map(income_map_norm)

    # For the dropdown we use the *display* strings, but map them back to codes
    display_cats = sorted(income_raw.unique())
    display_to_code = {}
    for disp in display_cats:
        norm = disp.strip().lower()
        display_to_code[disp] = income_map_norm.get(norm, 0)

    # ----- Confidence -----
    confidence_map = {
        "very confident": 3,
        "somewhat confident": 2,
        "unsure": 1,
        "not confident": 0,
    }
    df["Confidence"] = df["Confidence"].astype(str).str.lower()
    df["Confidence_code"] = df["Confidence"].map(confidence_map).fillna(1)

    # ----- Frequency of attendance -----
    attendance_map = {
        "always (0 - 1 absence per month)": 3,
        "frequently (2 - 4 absences per month)": 2,
        "occasionally (5 - 8 absences per month)": 1,
        "rarely (more than 8 absences per month)": 0,
    }
    df["Frequency of attendance"] = df["Frequency of attendance"].astype(str).str.lower()
    df["Attendance_code"] = df["Frequency of attendance"].map(attendance_map).fillna(1)

    # ----- Punctuality -----
    punctuality_map = {
        "always on time": 3,
        "occasionally late": 2,
        "frequently late": 1,
        "rarely on time": 0,
    }
    df["Punctuality"] = df["Punctuality"].astype(str).str.lower()
    df["Punctuality_code"] = df["Punctuality"].map(punctuality_map).fillna(1)

    # ----- Class engagement -----
    engagement_map = {
        "very engaged": 3,
        "moderately engaged": 2,
        "slightly engaged": 1,
        "not engaged": 0,
    }
    df["Class engagement"] = df["Class engagement"].astype(str).str.lower()
    df["Engagement_code"] = df["Class engagement"].map(engagement_map).fillna(1)

    # ----- Frequency of stress -----
    stress_map = {
        "always": 3,
        "frequently": 2,
        "sometimes": 1,
        "rarely": 0,
    }
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
    y = df["target"]

    return X, y, feature_cols, display_cats, display_to_code


# -----------------------------
# Build all 9 models
# -----------------------------
def build_models():
    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=300),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
        )
    return models


# -----------------------------
# 1) Train & compare all models (tab 1)
# -----------------------------
def train_and_compare_models(file_obj):
    if file_obj is None:
        return (
            "Please upload your survey CSV first.",
            None,
            None,
            "No report yet.",
            gr.update(choices=[], value=None),
        )

    # Load CSV (works both locally & on Render)
    try:
        df = pd.read_csv(file_obj.name)
    except Exception:
        file_obj.seek(0)
        df = pd.read_csv(file_obj)

    X, y, feat_cols, income_cats, income_map = preprocess_df(df)

    # 60/20/20 split (stratified), same as in thesis
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    models = build_models()

    rows = []
    lda_cm_img = None
    lda_report = None
    lda_model = None

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            val_acc = accuracy_score(y_val, y_val_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            rows.append([name, val_acc, test_acc])

            # Save extra details for LDA (best model in thesis)
            if name == "LDA":
                cm = confusion_matrix(y_test, y_test_pred)
                lda_cm_img = plot_confusion_matrix(cm, "Test Confusion Matrix — LDA")
                lda_report = classification_report(y_test, y_test_pred, zero_division=0)
                lda_model = model
        except Exception:
            # If a model fails (e.g., xgboost not installed properly), mark as NaN
            rows.append([name, np.nan, np.nan])

    summary_df = pd.DataFrame(rows, columns=["Model", "Validation Accuracy", "Test Accuracy"])
    summary_df = summary_df.sort_values("Test Accuracy", ascending=False)

    # Bar chart of Test Accuracy for all models
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary_df["Model"], summary_df["Test Accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison — Test Accuracy")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    bar_img = fig_to_pil(fig)

    # Markdown summary + table
    summary_md = (
        f"### Train / Validation / Test Split\n"
        f"- Train size: **{len(X_train)}**\n"
        f"- Validation size: **{len(X_val)}**\n"
        f"- Test size: **{len(X_test)}**\n\n"
        "### Model Comparison (Validation / Test Accuracy)\n"
        + summary_df.to_markdown(index=False)
    )

    if lda_report is None:
        lda_report_md = "LDA failed to train."
    else:
        lda_report_md = "### LDA Test Classification Report\n```text\n" + lda_report + "\n```"

    # Save LDA + encoding info for the prediction tab
    trained_artifacts["lda_model"] = lda_model
    trained_artifacts["feature_cols"] = feat_cols
    trained_artifacts["income_categories"] = income_cats
    trained_artifacts["income_map"] = income_map

    # Update the income dropdown with values from dataset
    income_dd_update = gr.update(
        choices=income_cats,
        value=(income_cats[0] if income_cats else None),
    )

    return summary_md, bar_img, lda_cm_img, lda_report_md, income_dd_update


# -----------------------------
# 2) Predict single student using LDA only (tab 2)
# -----------------------------
CONF_OPTIONS = [
    "Very confident",
    "Somewhat confident",
    "Unsure",
    "Not confident",
]

ATTEND_OPTIONS = [
    "Always (0 - 1 absence per month)",
    "Frequently (2 - 4 absences per month)",
    "Occasionally (5 - 8 absences per month)",
    "Rarely (more than 8 absences per month)",
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

STRESS_OPTIONS = [
    "Always",
    "Frequently",
    "Sometimes",
    "Rarely",
]


def predict_single(
    age,
    study_hours_choice,
    income_cat,
    confidence,
    attendance,
    punctuality,
    engagement,
    stress,
):
    if trained_artifacts["lda_model"] is None or trained_artifacts["income_map"] is None:
        return "⚠️ Please upload your CSV and run training in the first tab first."

    lda = trained_artifacts["lda_model"]
    feat_cols = trained_artifacts["feature_cols"]
    income_map = trained_artifacts["income_map"]

    if not income_cat:
        return "⚠️ Please select a family monthly income value from the dropdown."

    if income_cat not in income_map:
        return "❌ Income category not recognized. Please select one from the dropdown."

    # Age
    try:
        age_val = float(age)
    except Exception:
        return "❌ Age must be a number."

    # Study hours
    study_num = study_hours_to_num(study_hours_choice)

    # Income code from mapping created during preprocessing
    income_code = income_map[income_cat]

    confidence_map = {
        "very confident": 3,
        "somewhat confident": 2,
        "unsure": 1,
        "not confident": 0,
    }
    attendance_map = {
        "always (0 - 1 absence per month)": 3,
        "frequently (2 - 4 absences per month)": 2,
        "occasionally (5 - 8 absences per month)": 1,
        "rarely (more than 8 absences per month)": 0,
    }
    punctuality_map = {
        "always on time": 3,
        "occasionally late": 2,
        "frequently late": 1,
        "rarely on time": 0,
    }
    engagement_map = {
        "very engaged": 3,
        "moderately engaged": 2,
        "slightly engaged": 1,
        "not engaged": 0,
    }
    stress_map = {
        "always": 3,
        "frequently": 2,
        "sometimes": 1,
        "rarely": 0,
    }

    conf_code = confidence_map[confidence.lower()]
    attend_code = attendance_map[attendance.lower()]
    punct_code = punctuality_map[punctuality.lower()]
    engage_code = engagement_map[engagement.lower()]
    stress_code = stress_map[stress.lower()]

    row = pd.DataFrame(
        [{
            "Age": age_val,
            "Study_hours": study_num,
            "Income_code": income_code,
            "Confidence_code": conf_code,
            "Attendance_code": attend_code,
            "Punctuality_code": punct_code,
            "Engagement_code": engage_code,
            "Stress_code": stress_code,
        }]
    )[feat_cols]  # ensure same column order as training

    pred = lda.predict(row)[0]
    prob = None
    if hasattr(lda, "predict_proba"):
        prob = float(lda.predict_proba(row)[0, 1])  # probability of Success

    label = (
        "✅ Predicted: **Success (Good/Excellent)**"
        if pred == 1
        else "⚠️ Predicted: **At Risk / Below Good**"
    )
    if prob is not None:
        label += f"\n\nEstimated probability of Success class: **{prob:.2%}**"

    return label


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Thesis Model Dashboard") as demo:
    gr.Markdown("## 🎓 Thesis Model Dashboard — UC Student Performance Prediction")

    # TAB 1: training + comparison of 9 models
    with gr.Tab("1️⃣ Train & Compare Models"):
        file_in = gr.File(label="Upload survey dataset CSV", file_types=[".csv"])
        train_btn = gr.Button("Run Training & Evaluation", variant="primary")

        summary_md = gr.Markdown()
        bar_img = gr.Image(label="Model Comparison — Test Accuracy")
        lda_cm_img = gr.Image(label="LDA Test Confusion Matrix")
        lda_report_md = gr.Markdown()

    # TAB 2: prediction system (LDA only)
    with gr.Tab("2️⃣ Predict Single Student (LDA System)"):
        gr.Markdown(
            "Use the trained **LDA** model (identified as the best performer) "
            "to predict one student's academic performance."
        )

        with gr.Row():
            age_in = gr.Number(label="Age", value=18)
            study_hours_in = gr.Dropdown(
                label="Study hours per week (category)",
                choices=[
                    "Less than 5 hours",
                    "5 - 10 hours",
                    "11 - 15 hours",
                    "More than 15 hours",
                ],
                value="5 - 10 hours",
            )

        income_dd = gr.Dropdown(
            label="Family monthly income (from uploaded CSV)",
            choices=[],
            value=None,
            interactive=True,
        )

        with gr.Row():
            conf_dd = gr.Dropdown(label="Confidence", choices=CONF_OPTIONS, value=CONF_OPTIONS[1])
            attend_dd = gr.Dropdown(
                label="Frequency of attendance",
                choices=ATTEND_OPTIONS,
                value=ATTEND_OPTIONS[0],
            )

        with gr.Row():
            punct_dd = gr.Dropdown(label="Punctuality", choices=PUNCT_OPTIONS, value=PUNCT_OPTIONS[0])
            engage_dd = gr.Dropdown(
                label="Class engagement",
                choices=ENGAGE_OPTIONS,
                value=ENGAGE_OPTIONS[1],
            )

        stress_dd = gr.Dropdown(
            label="Frequency of stress",
            choices=STRESS_OPTIONS,
            value=STRESS_OPTIONS[2],
        )

        predict_btn = gr.Button("Predict Student Performance", variant="primary")
        pred_out = gr.Markdown()

    # Wire up
    train_btn.click(
        train_and_compare_models,
        inputs=[file_in],
        outputs=[summary_md, bar_img, lda_cm_img, lda_report_md, income_dd],
    )

    predict_btn.click(
        predict_single,
        inputs=[
            age_in,
            study_hours_in,
            income_dd,
            conf_dd,
            attend_dd,
            punct_dd,
            engage_dd,
            stress_dd,
        ],
        outputs=[pred_out],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
