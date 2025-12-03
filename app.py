import warnings
warnings.filterwarnings("ignore")

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

# 9 Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

trained = {
    "lda": None,
    "feature_cols": None,
    "income_cats": None,
    "income_map": None,
    "models": None
}

###############################################
# Helpers
###############################################

def fig_to_pil(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def plot_cm(cm, title):
    fig, ax = plt.subplots(figsize=(4.5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Fail","Success"])
    ax.set_yticklabels(["Fail","Success"])
    for i in range(2):
        for j in range(2):
            ax.text(j,i,cm[i,j],ha="center",va="center")
    fig.tight_layout()
    return fig_to_pil(fig)

###############################################
# Preprocessing
###############################################

EXPECTED = [
    "Age","Study hours per week","Family monthly income","Confidence",
    "Frequency of attendance","Punctuality","Class engagement",
    "Frequency of stress","Performance"
]

def study_to_num(x):
    if pd.isna(x): return 0
    t=str(x).lower()
    if "less than 5" in t: return 3
    if "5 - 10" in t: return 7.5
    if "11 - 15" in t: return 13
    if "more than 15" in t: return 16
    try: return float(t)
    except: return 0

def preprocess(df_raw):
    df=df_raw.copy()
    df.columns=df.columns.str.strip()

    miss=[c for c in EXPECTED if c not in df.columns]
    if miss:
        raise ValueError("Missing: "+", ".join(miss))

    # Target
    def mapperf(p):
        if isinstance(p,str) and ("good" in p.lower() or "excellent" in p.lower()):
            return 1
        return 0

    df["target"]=df["Performance"].apply(mapperf)

    df["Age"]=pd.to_numeric(df["Age"],errors="coerce").fillna(df["Age"].median())

    df["Study_hours"]=df["Study hours per week"].apply(study_to_num)

    # Family income
    raw=df["Family monthly income"].astype(str).str.replace("\xa0"," ").str.strip()
    norm=raw.str.lower()
    uniq=sorted(norm.unique())
    norm_map={cat:i for i,cat in enumerate(uniq)}
    df["Income_code"]=norm.map(norm_map)

    disp = sorted(raw.unique())
    disp_map={}
    for d in disp:
        disp_map[d]=norm_map[d.lower()]

    # Ordinal mappings
    conf_map={"very confident":3,"somewhat confident":2,"unsure":1,"not confident":0}
    att_map={
        "always (0 - 1 absence per month)":3,
        "frequently (2 - 4 absences per month)":2,
        "sometimes (5 - 7 absences per month)":1,
        "rarely (more than 7 absences per month)":0
    }
    pun_map={
        "always on time":3,"occasionally late":2,"frequently late":1,"rarely on time":0
    }
    eng_map={
        "very engaged":3,"moderately engaged":2,"slightly engaged":1,"not engaged":0
    }
    str_map={
        "always":3,"frequently":2,"sometimes":1,"rarely":0
    }

    df["Confidence_code"]=df["Confidence"].str.lower().map(conf_map).fillna(1)
    df["Attendance_code"]=df["Frequency of attendance"].str.lower().map(att_map).fillna(1)
    df["Punctuality_code"]=df["Punctuality"].str.lower().map(pun_map).fillna(1)
    df["Engagement_code"]=df["Class engagement"].str.lower().map(eng_map).fillna(1)
    df["Stress_code"]=df["Frequency of stress"].str.lower().map(str_map).fillna(1)

    feats=[
        "Age","Study_hours","Income_code","Confidence_code","Attendance_code",
        "Punctuality_code","Engagement_code","Stress_code"
    ]

    return df[feats], df["target"], feats, disp, disp_map

###############################################
# Build 9 models
###############################################

def build_models():
    m=[
        (LinearDiscriminantAnalysis(),"LDA"),
        (AdaBoostClassifier(random_state=42),"AdaBoost"),
        (xgb.XGBClassifier(use_label_encoder=False,eval_metric="logloss",random_state=42),"XGBoost") if HAS_XGB else (None,"XGBoost"),
        (RandomForestClassifier(n_estimators=300,random_state=42),"RandomForest"),
        (LogisticRegression(max_iter=1500,random_state=42),"LogisticRegression"),
        (GaussianNB(),"NaiveBayes"),
        (DecisionTreeClassifier(random_state=42),"DecisionTree"),
        (SVC(probability=True,random_state=42),"SVM"),
        (KNeighborsClassifier(),"KNN")
    ]
    if not HAS_XGB:
        m=[i for i in m if i[1]!="XGBoost"]
    return m

###############################################
# Training + Comparison (Tab 1)
###############################################

def train_compare(file):
    if file is None:
        return ("Upload CSV",None,None,"No report",gr.update(),gr.update(),{})

    try:
        df=pd.read_csv(file.name)
    except:
        file.seek(0); df=pd.read_csv(file)

    X,y,feats,incats,inmap=preprocess(df)

    Xtr,Xtemp,ytr,ytemp=train_test_split(X,y,test_size=0.4,stratify=y,random_state=42)
    Xv,Xte,yv,yte=train_test_split(Xtemp,ytemp,test_size=0.5,stratify=ytemp,random_state=42)

    models=build_models()
    rows=[]
    details={}
    lda_cm_test=None
    lda_report=None
    lda_model=None

    for model,name in models:
        if model is None:
            rows.append([name,np.nan,np.nan])
            continue

        model.fit(Xtr,ytr)
        yvp=model.predict(Xv)
        ytp=model.predict(Xte)

        va=accuracy_score(yv,yvp)
        ta=accuracy_score(yte,ytp)
        rows.append([name,va,ta])

        cmv=plot_cm(confusion_matrix(yv,yvp),f"Validation CM — {name}")
        cmt=plot_cm(confusion_matrix(yte,ytp),f"Test CM — {name}")
        rep=classification_report(yte,ytp,zero_division=0)

        details[name]={
            "cmv":cmv,
            "cmt":cmt,
            "rep":rep
        }

        if name=="LDA":
            lda_cm_test=cmt
            lda_report=rep
            lda_model=model

    dfsum=pd.DataFrame(rows,columns=["Model","Val Accuracy","Test Accuracy"])
    dfsum=dfsum.sort_values("Test Accuracy",ascending=False)

    # Line graph
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(dfsum["Model"],dfsum["Test Accuracy"],marker="o")
    ax.set_ylim(0,1)
    ax.tick_params(axis="x",rotation=45)
    ax.set_title("Test Accuracy (Line Graph)")
    line=fig_to_pil(fig)

    # Markdown summary
    summ=(
        f"### Dataset Split\n"
        f"- Train: **{len(Xtr)}**\n- Val: **{len(Xv)}**\n- Test: **{len(Xte)}**\n\n"
        "### Accuracy Table\n"
        +dfsum.to_markdown(index=False)
    )

    # LDA report
    lda_md="```text\n"+lda_report+"\n```" if lda_report else "LDA failed."

    # Save artifacts for prediction tab
    trained["lda"]=lda_model
    trained["feature_cols"]=feats
    trained["income_cats"]=incats
    trained["income_map"]=inmap

    income_update=gr.update(choices=incats,value=incats[0] if incats else None)
    model_update=gr.update(choices=dfsum["Model"].tolist(),
                           value=dfsum["Model"].iloc[0] if len(dfsum)>0 else None)

    return summ, line, lda_cm_test, lda_md, income_update, model_update, details
###############################################
# LDA Prediction (Tab 2)
###############################################

CONF_OPTS = [
    "Very confident","Somewhat confident","Unsure","Not confident"
]
ATT_OPTS = [
    "Always (0 - 1 absence per month)",
    "Frequently (2 - 4 absences per month)",
    "Sometimes (5 - 7 absences per month)",
    "Rarely (more than 7 absences per month)"
]
PUN_OPTS = [
    "Always on time","Occasionally late","Frequently late","Rarely on time"
]
ENG_OPTS = [
    "Very engaged","Moderately engaged","Slightly engaged","Not engaged"
]
STRESS_OPTS = ["Always","Frequently","Sometimes","Rarely"]

def predict_single(
    age, study_hours, income_cat, confidence, attendance,
    punctuality, engagement, stress
):
    lda = trained["lda"]
    feats = trained["feature_cols"]
    imap = trained["income_map"]

    if lda is None:
        return "⚠️ Please run training first in Tab 1."

    if income_cat not in imap:
        return "⚠️ Invalid income selection."

    try:
        age_val = float(age)
    except:
        return "❌ Age must be numeric."

    # Same encodings used during training
    study_num = study_to_num(study_hours)
    inc_code = imap[income_cat]

    conf_map={"very confident":3,"somewhat confident":2,"unsure":1,"not confident":0}
    att_map={
        "always (0 - 1 absence per month)":3,
        "frequently (2 - 4 absences per month)":2,
        "sometimes (5 - 7 absences per month)":1,
        "rarely (more than 7 absences per month)":0
    }
    pun_map={"always on time":3,"occasionally late":2,"frequently late":1,"rarely on time":0}
    eng_map={"very engaged":3,"moderately engaged":2,"slightly engaged":1,"not engaged":0}
    str_map={"always":3,"frequently":2,"sometimes":1,"rarely":0}

    row = pd.DataFrame([{
        "Age": age_val,
        "Study_hours": study_num,
        "Income_code": inc_code,
        "Confidence_code": conf_map[confidence.lower()],
        "Attendance_code": att_map[attendance.lower()],
        "Punctuality_code": pun_map[punctuality.lower()],
        "Engagement_code": eng_map[engagement.lower()],
        "Stress_code": str_map[stress.lower()],
    }])[feats]

    # LDA Probabilities
    proba = lda.predict_proba(row)[0]
    prob_fail = float(proba[0])
    prob_success = float(proba[1])
    pred = 1 if prob_success >= 0.5 else 0

    # Main label
    if pred == 1:
        main = "✅ **Predicted: Success (Good/Excellent)**"
    else:
        main = "⚠️ **Predicted: At Risk / Below Good**"

    # Risk wording
    if prob_success >= 0.6:
        risk = "**Risk Level:** Likely Success"
    elif prob_success <= 0.4:
        risk = "**Risk Level:** Likely At Risk"
    else:
        risk = "**Risk Level:** Borderline / Uncertain"

    details = (
        f"\n\n### Probability Breakdown\n"
        f"- Success (Good/Excellent): **{prob_success:.2%}**\n"
        f"- At Risk / Below Good: **{prob_fail:.2%}**"
    )

    return main + "\n\n" + risk + details

###############################################
# Per-model dropdown renderer
###############################################

def show_details(name, store):
    if not name or name not in store:
        return None, None, "Select a model to view details."
    d = store[name]
    md = "```text\n"+d["rep"]+"\n```"
    return d["cmv"], d["cmt"], md

###############################################
# Gradio UI
###############################################

with gr.Blocks(title="Thesis Model Dashboard") as demo:

    gr.Markdown("## 🎓 Thesis Model Dashboard — Predicting Student Academic Performance")

    store = gr.State({})

    ###############################################
    # TAB 1 — Train & Compare
    ###############################################
    with gr.Tab("1️⃣ Train & Compare Models"):
        file_in = gr.File(label="Upload CSV", file_types=[".csv"])
        run_btn = gr.Button("Run Training", variant="primary")

        summary_md = gr.Markdown()
        line_img = gr.Image(label="Model Comparison (Line Graph)")
        lda_cm_img = gr.Image(label="LDA Test Confusion Matrix")
        lda_rep_md = gr.Markdown()

        gr.Markdown("### 🔍 Per-Model Detailed Results")
        model_select = gr.Dropdown(label="Select model", choices=[], value=None)

        with gr.Tabs():
            with gr.Tab("Validation Confusion Matrix"):
                cm_val_out = gr.Image()
            with gr.Tab("Test Confusion Matrix"):
                cm_test_out = gr.Image()
            with gr.Tab("Test Classification Report"):
                rep_out = gr.Markdown()

    ###############################################
    # TAB 2 — LDA Prediction System
    ###############################################
    with gr.Tab("2️⃣ Predict Single Student (LDA)"):
        gr.Markdown("Use LDA (identified as the best model) to predict one student's outcome.")

        with gr.Row():
            age_in = gr.Number(label="Age", value=18)
            study_in = gr.Dropdown(
                label="Study hours per week",
                choices=[
                    "Less than 5 hours","5 - 10 hours",
                    "11 - 15 hours","More than 15 hours"
                ],
                value="5 - 10 hours"
            )

        income_in = gr.Dropdown(label="Family monthly income", choices=[], value=None)

        with gr.Row():
            conf_in = gr.Dropdown(label="Confidence", choices=CONF_OPTS, value="Somewhat confident")
            att_in = gr.Dropdown(label="Attendance", choices=ATT_OPTS, value=ATT_OPTS[0])

        with gr.Row():
            pun_in = gr.Dropdown(label="Punctuality", choices=PUN_OPTS, value=PUN_OPTS[0])
            eng_in = gr.Dropdown(label="Engagement", choices=ENG_OPTS, value=ENG_OPTS[1])

        stress_in = gr.Dropdown(label="Stress", choices=STRESS_OPTS, value="Sometimes")

        pred_btn = gr.Button("Predict", variant="primary")
        pred_out = gr.Markdown()

    ###############################################
    # Actions
    ###############################################

    run_btn.click(
        train_compare,
        inputs=[file_in],
        outputs=[
            summary_md,
            line_img,
            lda_cm_img,
            lda_rep_md,
            income_in,
            model_select,
            store
        ]
    )

    model_select.change(
        show_details,
        inputs=[model_select, store],
        outputs=[cm_val_out, cm_test_out, rep_out]
    )

    pred_btn.click(
        predict_single,
        inputs=[
            age_in, study_in, income_in, conf_in, att_in,
            pun_in, eng_in, stress_in
        ],
        outputs=[pred_out]
    )

###############################################
# Launch
###############################################

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
