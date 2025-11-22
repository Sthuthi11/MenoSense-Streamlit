import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# =========================
# CONFIG
# =========================
DATA_PATH = "full_dataset.csv"
TARGET_COL = "Menopause status"

# =========================
# MANUAL CATEGORY MAPPINGS
# =========================
age_map = {
    "30-40": 1, "40-45": 2, "45-50": 3, "50-55": 4,
    "55-60": 5, "60-70": 6, "Above 70": 7, "15-30": 8
}

profession_map = {
    "Housewife": 1, "IT": 2, "Lecturer": 3,
    "Part-time worker": 4, "Private shops and business": 5,
    "Doctor": 6, "Employee": 7, "Other (Student)": 8
}

children_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4 or more": 4}
menarche_map = {"Below 11": 1, "11-13": 2, "14-18": 3}

cycle_map = {
    "Regular": 1, "Irregular": 2,
    "Skipped periods for a few months": 3,
    "No periods for more than 12 months": 4
}

flow_map = {
    "No change": 1, "Heavier than usual": 2,
    "Lighter than usual": 3, "Spotting between periods": 4,
    "Menopause": 5
}

blood_map = {
    "No change": 1,
    "Darker than usual (brown/black)": 2,
    "Darker discharge color (red)": 3
}

family_history_map = {"Yes": 1, "No": 0, "Not sure": 2}
surgery_map = {"Yes": 1, "No": 0}
hormone_map = {"Yes": 1, "No": 0}
period_delay_map = {"Yes, frequently": 1, "Yes, occasionally": 2, "No": 0}
copper_t_map = {"Yes": 1, "No": 0}
pain_med_map = {"Yes, frequently": 1, "Yes, occasionally": 2, "No": 0}

activity_map = {
    "Daily": 1, "A few times a week": 2,
    "Rarely": 3, "Never": 4
}

menopause_map = {
    "Yes, 30-45 years": 1, "Yes, 45-50 years": 1,
    "Yes, 50-60 years": 1, "Yes, 60-65 years": 1,
    "Yes, above 65 years": 1, "No": -1
}

# =========================
# SYMPTOMS LIST
# =========================
symptoms_list = [
    "Hot flashes", "Night sweats", "Difficulty getting to sleep",
    "Fatigue or feeling dizzy", "Heart palpitations or a sensation of the heart skipping beats",
    "Anxiety", "Mood swings", "Frequent headaches or migraines",
    "Difficulty concentrating or memory issues",
    "Pain or burning when urinating", "Bladder infections",
    "Uncomfortable gas or bloating", "Vaginal dryness or discomfort",
    "Itching or abnormal vaginal discharge", "Reduced libido",
    "Increased hair loss or thinning", "Hair growth on the face or body",
    "Skin changes (dryness, acne, or itchiness)",
    "Weight gain or metabolism changes",
    "Joint or muscle pain"
]

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df["Age"] = df["Age"].replace(age_map)
    df["Profession"] = df["Profession"].replace(profession_map)
    df["Number of children"] = df["Number of children"].replace(children_map)
    df["Menarche age"] = df["Menarche age"].replace(menarche_map)
    df["Cycle regularity"] = df["Cycle regularity"].replace(cycle_map)
    df["Flow change"] = df["Flow change"].replace(flow_map)
    df["Blood color change"] = df["Blood color change"].replace(blood_map)
    df["Family history"] = df["Family history"].replace(family_history_map)
    df["Surgeries"] = df["Surgeries"].replace(surgery_map)
    df["Hormone condition"] = df["Hormone condition"].replace(hormone_map)
    df["Period delay meds"] = df["Period delay meds"].replace(period_delay_map)
    df["Copper-T"] = df["Copper-T"].replace(copper_t_map)
    df["Pain medication"] = df["Pain medication"].replace(pain_med_map)
    df["Physical activity"] = df["Physical activity"].replace(activity_map)

    df["Menopause status"] = df["Menopause status"].replace(menopause_map)
    df["Menopause status"] = df["Menopause status"].replace({"Yes": 1, "No": -1})
    df["Menopause status"] = pd.to_numeric(df["Menopause status"], errors="coerce").fillna(-1)

    for symptom in symptoms_list:
        df[symptom] = -1

    if "symptom" in df.columns:
        for i, row in df.iterrows():
            if pd.notna(row["symptom"]):
                found = [s.strip() for s in row["symptom"].split(",")]
                for s in found:
                    if s in symptoms_list:
                        df.at[i, s] = 1
        df.drop(columns=["symptom"], inplace=True)

    return df

# =========================
# TRAIN MODEL
# =========================
@st.cache_resource
def train_model():
    df = preprocess_data(pd.read_csv(DATA_PATH))

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    X_test = scaler.transform(X_test)

    base_estimators = [
        ("ada", AdaBoostClassifier(random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
        ("svm", SVC(probability=True, kernel="rbf")),
    ]

    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5
    )

    model.fit(X_res, y_res)
    acc = accuracy_score(y_test, model.predict(X_test))

    return model, scaler, feature_cols, acc

# =========================
# STAGE INTERPRETATION
# =========================
def map_stage(pred, age_code):
    if pred == -1:
        return "Premenopause"
    if pred == 1 and age_code <= 3:
        return "Perimenopause"
    return "Postmenopause"

stage_steps = {
    "Premenopause": [
        "Maintain a healthy lifestyle.",
        "Track menstrual cycle changes.",
        "Consult a gynecologist for unusual symptoms."
    ],
    "Perimenopause": [
        "Monitor symptoms like hot flashes and mood changes.",
        "Practice stress reduction and sleep hygiene.",
        "Discuss treatment options with a gynecologist."
    ],
    "Postmenopause": [
        "Get bone density and heart health checkups.",
        "Maintain exercise and a nutritious diet.",
        "Follow preventive health guidelines."
    ]
}

# =========================
# UI HELPERS
# =========================
def yesno(v): return 1 if v == "Yes" else -1

def build_user_input(feature_cols):
    st.subheader("Enter Details")

    c1, c2 = st.columns(2)

    with c1:
        age = st.selectbox("Age group", ["Select"] + list(age_map.keys()))
        prof = st.selectbox("Profession", ["Select"] + list(profession_map.keys()))
        child = st.selectbox("Number of children", ["Select"] + list(children_map.keys()))
        menarche = st.selectbox("Menarche age", ["Select"] + list(menarche_map.keys()))

    with c2:
        cycle = st.selectbox("Cycle regularity", ["Select"] + list(cycle_map.keys()))
        flow = st.selectbox("Flow change", ["Select"] + list(flow_map.keys()))
        blood = st.selectbox("Blood color change", ["Select"] + list(blood_map.keys()))
        family = st.selectbox("Family history", ["Select"] + list(family_history_map.keys()))

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        surgery = st.radio("Surgeries", ["No", "Yes"])
        hormone = st.radio("Hormone condition", ["No", "Yes"])
        copper = st.radio("Used Copper-T?", ["No", "Yes"])

    with col4:
        period_med = st.selectbox("Medicine to delay periods?", ["Select"] + list(period_delay_map.keys()))
        pain_med = st.selectbox("Pain medication use?", ["Select"] + list(pain_med_map.keys()))
        activity = st.selectbox("Physical activity", ["Select"] + list(activity_map.keys()))

    st.markdown("---")
    st.subheader("Symptoms")

    sym_vals = {}
    cols = st.columns(2)
    for i, s in enumerate(symptoms_list):
        sym_vals[s] = yesno(cols[i % 2].radio(s, ["No", "Yes"]))

    mandatory = {
        "Age": age_map.get(age),
        "Profession": profession_map.get(prof),
        "Number of children": children_map.get(child),
        "Menarche age": menarche_map.get(menarche),
        "Cycle": cycle_map.get(cycle),
        "Flow": flow_map.get(flow),
        "Blood": blood_map.get(blood),
        "Family": family_history_map.get(family),
        "Period": period_delay_map.get(period_med),
        "Pain": pain_med_map.get(pain_med),
        "Activity": activity_map.get(activity),
    }

    if any(v is None for v in mandatory.values()):
        st.warning("Please fill all dropdowns.")
        return None

    data = {
        "Age": mandatory["Age"],
        "Profession": mandatory["Profession"],
        "Number of children": mandatory["Number of children"],
        "Menarche age": mandatory["Menarche age"],
        "Cycle regularity": mandatory["Cycle"],
        "Flow change": mandatory["Flow"],
        "Blood color change": mandatory["Blood"],
        "Family history": mandatory["Family"],
        "Surgeries": surgery_map[surgery],
        "Hormone condition": hormone_map[hormone],
        "Period delay meds": mandatory["Period"],
        "Copper-T": copper_t_map[copper],
        "Pain medication": mandatory["Pain"],
        "Physical activity": mandatory["Activity"],
    }

    data.update(sym_vals)

    return pd.DataFrame([data])[feature_cols]

# =========================
# MAIN
# =========================
def main():
    st.title("MenoSense – Menopause Stage Prediction")
    st.write(
        "This app uses a stacked ensemble model (KNN, AdaBoost, SVM) "
        "to predict a woman's menopause stage based on her symptoms and medical history."
    )

    try:
        model, scaler, feature_cols, acc = train_model()
        st.success(f"Model loaded (Accuracy: {acc:.2f})")
    except Exception as e:
        st.error("Error loading dataset or training model.")
        st.write(e)
        return

    user_df = build_user_input(feature_cols)

    if st.button("Predict Menopause Stage"):
        if user_df is None:
            st.error("Please complete all fields.")
            return

        scaled = scaler.transform(user_df.values)
        pred = int(model.predict(scaled)[0])
        age_code = int(user_df["Age"][0])

        stage = map_stage(pred, age_code)

        st.subheader("Prediction Result")
        st.success(f"Predicted Stage: **{stage}**")

        st.subheader("Recommended Steps")
        for s in stage_steps[stage]:
            st.markdown(f"- {s}")

if __name__ == "__main__":
    main()
