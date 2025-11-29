import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image

import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# -----------------------------
# Globals to hold trained model
# -----------------------------
trained_artifacts = {
    "model": None,
    "feature_cols": None,
    "income_le": None,
    "income_choices": [],
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
# Preprocessing (matches thesis)
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

    # Target mapping: Good/Excellent -> 1, others -> 0
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

    # Family monthly income -> clean + label encode
    df["Family monthly income"] = (
        df["Family monthly income"]
        .astype(str)
        .str.replace("\xa0", " ")
        .str.strip()
        .str.lower()
    )
    income_le = LabelEncoder()
    df["Income_code"] = income_le.fit_transform(df["Family monthly income"])
    income_choices = sorted(df["Family monthly income"].unique())

    # Confidence
    confidence_map = {
        "very confident": 3,
        "somewhat confident": 2,
        "unsure": 1,
        "not confident": 0,
    }
    df["Confidence"] = df["Confidence"].astype(str).str.lower()
    df["Confidence_code"] = df["Confidence"].map(confidence_map).fillna(1)

    # Frequency of attendance
    attendance_map = {
        "always (0 - 1 absence per month)": 3,
        "frequently (2 - 4 absences per month)": 2,
        "occasionally (5 - 8 absences per month)": 1,
        "rarely (more than 8 absences per month)": 0,
    }
    df["Frequency of attendance"] = df["Frequency of attendance"].astype(str).str.lower()
    df["Attendance_code"] = df["Frequency of attendance"].map(attendance_map).fillna(1)

    # Punctuality
    punctuality_map = {
        "always on time": 3,
        "occasionally late": 2,
        "frequently late": 1,
        "rarely on time": 0,
    }
    df["Punctuality"] = df["Punctuality"].astype(str).str.lower()
    df["Punctuality_code"] = df["Punctuality"].map(punctuality_map).fillna(1)

    # Class engagement
    engagement_map = {
        "very engaged": 3,
        "moderately engaged": 2,
        "slightly engaged": 1,
        "not engaged": 0,
    }
    df["Class engagement"] = df["Class engagement"].astype(str).str.lower()
    df["Engagement_code"] = df["Class engagement"].map(engagement_map).fillna(1)

    # Frequency of stress
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

    return X, y, feature_cols, income_le, income_choices


# -----------------------------
# 1) Train & evaluate LDA
# -----------------------------
def train_lda(file_obj):
    if file_obj is None:
        return "Please upload your survey CSV first.", None, "No report yet."

    # Load CSV (Render/Gradio sometimes gives .name, sometimes file-like)
    try:
        df = pd.read_csv(file_obj.name)
    except Exception:
        file_obj.seek(0)
        df = pd.read_csv(file_obj)

    try:
        X, y, feat_cols, income_le, income_choices = preprocess_df(df)
    except Exception as e:
        return f"❌ Preprocessing error: {e}", None, "No report due to preprocessing error."

    # 60/20/20 stratified split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # Train LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Evaluate
    y_val_pred = lda.predict(X_val)
    y_test_pred = lda.predict(X_test)

    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    cm_test = confusion_matrix(y_test, y_test_pred)
    cm_img = plot_confusion_matrix(cm_test, "Test Confusion Matrix — LDA")

    report = classification_report(y_test, y_test_pred, zero_division=0)

    summary_md = (
        f"### LDA Training Summary\n"
        f"- Train size: **{len(X_train)}**\n"
        f"- Validation size: **{len(X_val)}**\n"
        f"- Test size: **{len(X_test)}**\n"
        f"- Validation Accuracy: **{val_acc:.4f}**\n"
        f"- Test Accuracy: **{test_acc:.4f}**\n"
    )

    report_md = "### Test Classification Report\n```text\n" + report + "\n```"

    # Save to global state for prediction tab
    trained_artifacts["model"] = lda
    trained_artifacts["feature_cols"] = feat_cols
    trained_artifacts["income_le"] = income_le
    trained_artifacts["income_choices"] = income_choices

    # Update income dropdown choices for prediction
    income_dropdown_update = gr.update(choices=income_choices,
                                       value=income_choices[0] if income_choices else None)

    return summary_md, cm_img, report_md, income_dropdown_update


# -----------------------------
# 2) Predict single student
# -----------------------------
# Use same category labels as in survey
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
    if trained_artifacts["model"] is None:
        return "⚠️ Please upload your CSV and train the LDA model first in the other tab."

    lda = trained_artifacts["model"]
    feat_cols = trained_artifacts["feature_cols"]
    income_le = trained_artifacts["income_le"]

    # Transform inputs
    age_val = float(age)

    study_num = study_hours_to_num(study_hours_choice)

    income_proc = str(income_cat).strip().lower()
    try:
        income_code = int(income_le.transform([income_proc])[0])
    except Exception:
        return "❌ Income category not recognized. Make sure it exists in the uploaded CSV."

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
    )[feat_cols]  # ensure same column order

    # Prediction
    pred = lda.predict(row)[0]
    if hasattr(lda, "predict_proba"):
        prob = lda.predict_proba(row)[0, 1]  # probability of Success
    else:
        prob = np.nan

    label = "✅ Predicted: **Success (Good/Excellent)**" if pred == 1 else "⚠️ Predicted: **At Risk / Below Good**"

    if np.isnan(prob):
        prob_txt = ""
    else:
        prob_txt = f"\n\nEstimated probability of Success class: **{prob:.2%}**"

    return label + prob_txt


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="Thesis LDA Dashboard") as demo:
    gr.Markdown("## 🎓 Thesis LDA Dashboard — UC Student Performance Prediction")

    with gr.Tab("1️⃣ Train & Evaluate LDA"):
        file_in = gr.File(label="Upload survey dataset CSV", file_types=[".csv"])
        train_btn = gr.Button("Run LDA Training & Evaluation", variant="primary")

        summary_md = gr.Markdown()
        cm_img = gr.Image(label="Test Confusion Matrix (LDA)")
        report_md = gr.Markdown()

        # hidden output to update income dropdown in other tab
        income_dd_proxy = gr.Dropdown(visible=False)

        train_btn.click(
            train_lda,
            inputs=[file_in],
            outputs=[summary_md, cm_img, report_md, income_dd_proxy],
        )

    with gr.Tab("2️⃣ Predict Single Student (LDA System)"):
        gr.Markdown("Use the trained LDA model to predict **one student's** performance.")

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
            attend_dd = gr.Dropdown(label="Frequency of attendance", choices=ATTEND_OPTIONS,
                                    value=ATTEND_OPTIONS[0])

        with gr.Row():
            punct_dd = gr.Dropdown(label="Punctuality", choices=PUNCT_OPTIONS, value=PUNCT_OPTIONS[0])
            engage_dd = gr.Dropdown(label="Class engagement", choices=ENGAGE_OPTIONS,
                                    value=ENGAGE_OPTIONS[1])

        stress_dd = gr.Dropdown(label="Frequency of stress", choices=STRESS_OPTIONS,
                                value=STRESS_OPTIONS[2])

        predict_btn = gr.Button("Predict Student Performance", variant="primary")
        pred_out = gr.Markdown()

        # When we trained, we updated a hidden dropdown; mirror its choices here
        def sync_income_choices(hidden_dropdown_value, hidden_dropdown_choices):
            return gr.update(choices=hidden_dropdown_choices,
                             value=(hidden_dropdown_choices[0] if hidden_dropdown_choices else None))

        income_dd_proxy.change(
            sync_income_choices,
            inputs=[income_dd_proxy, income_dd_proxy],
            outputs=[income_dd],
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
