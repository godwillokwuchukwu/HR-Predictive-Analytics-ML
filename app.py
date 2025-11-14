# HR.py -- Fixed and robust Streamlit HR Analytics app

import os
import joblib
import pickle
import json
import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="HR Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM CSS =====================
st.markdown("""
<style>
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.08);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .kpi-label {
        font-size: 0.95rem;
        color: #555;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ================== LOAD DATA ==================
@st.cache_data
def load_data(path="hr_processed.csv"):
    if not os.path.exists(path):
        st.error(f"Data file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df

# ================== LOAD MODELS & ARTIFACTS =================
@st.cache_resource
def load_artifacts():
    """
    Load models, scalers, and label encoders used by the app.
    Returns (models_dict, scalers_dict, le_dict)
    """
    artifacts = {
        "models": {},
        "scalers": {},
        "le_dict": {}
    }

    def safe_joblib_load(path):
        if not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load {path}: {e}")
            return None

    # Common model/scaler filenames used in this app (keep consistent with train scripts)
    artifacts["models"]["attrition"] = safe_joblib_load("attr_model.pkl")
    artifacts["scalers"]["attrition"] = safe_joblib_load("scaler_attr.pkl")

    artifacts["models"]["performance"] = safe_joblib_load("perf_model.pkl")
    artifacts["models"]["salary"] = safe_joblib_load("sal_model.pkl")
    artifacts["scalers"]["salary"] = safe_joblib_load("sal_scaler.pkl")

    artifacts["models"]["kmeans"] = safe_joblib_load("kmeans_model.pkl")
    artifacts["scalers"]["cluster"] = safe_joblib_load("cluster_scaler.pkl")

    artifacts["models"]["promo"] = safe_joblib_load("promo_model.pkl")
    artifacts["scalers"]["promo"] = safe_joblib_load("promo_scaler.pkl")

    # other optional models
    artifacts["models"]["ot"] = safe_joblib_load("ot_model.pkl")
    artifacts["scalers"]["ot"] = safe_joblib_load("ot_scaler.pkl")

    artifacts["models"]["tenure"] = safe_joblib_load("tenure_model.pkl")
    artifacts["models"]["train_effect"] = safe_joblib_load("train_effect_model.pkl")

    # Label encoders dictionary
    le_path = "le_dict.pkl"
    if os.path.exists(le_path):
        try:
            with open(le_path, "rb") as f:
                artifacts["le_dict"] = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load label encoders ({le_path}): {e}")
            artifacts["le_dict"] = {}
    else:
        artifacts["le_dict"] = {}
        st.warning(f"Label encoder file not found: {le_path} — decoding/encoding will use fallback methods.")

    return artifacts["models"], artifacts["scalers"], artifacts["le_dict"]


# ---------- Load everything ----------
df = load_data()
models, scalers, le_dict = load_artifacts()

# Stop early if no dataset
if df.empty:
    st.stop()

# ================== SAFE ENCODING / DECODING HELPERS =============
def safe_encode(col, series, le_dict_local=le_dict):
    """
    Encode a pandas Series using label encoder from le_dict.
    Unknown values map to 'UNK' class index if available, otherwise -1.
    Returns numpy int array.
    """
    s = series.astype(str).fillna("UNK")
    if col not in le_dict_local:
        encoded, _ = pd.factorize(s)
        return np.array(encoded, dtype=int)

    le = le_dict_local[col]
    classes = list(le.classes_)
    mapping = {lab: idx for idx, lab in enumerate(classes)}
    unk_idx = mapping.get("UNK", None)

    enc = [mapping[v] if v in mapping else (unk_idx if unk_idx is not None else -1) for v in s.tolist()]
    return np.array(enc, dtype=int)

def safe_transform_single(col, value, le_dict_local=le_dict):
    """
    Safely transform a single categorical value using label encoder in le_dict.
    Returns integer label (or fallback -1).
    """
    if col not in le_dict_local:
        # fallback: factorize single value -> 0
        return 0
    le = le_dict_local[col]
    val = str(value)
    classes = list(le.classes_)
    if val in classes:
        return int(le.transform([val])[0])
    if "UNK" in classes:
        return int(le.transform(["UNK"])[0])
    # As last resort, extend classes to include UNK (non-persistent) then map
    try:
        # create mapping for fallback
        mapping = {lab: idx for idx, lab in enumerate(classes)}
        return int(mapping.get(val, -1))
    except Exception:
        return -1

def decode(col, val):
    """
    Safely decode numeric label to original category using le_dict.
    If fails, return the input.
    """
    if col not in le_dict:
        return val
    le = le_dict[col]
    try:
        return le.inverse_transform([int(val)])[0]
    except Exception:
        return val

def decoded_unique_options(col_name):
    """
    Return a list of unique decoded options for a column in the dataset,
    attempting to inverse transform when possible.
    """
    if col_name not in df.columns:
        return []
    uniques = df[col_name].dropna().unique().tolist()
    out = []
    for u in uniques:
        try:
            decoded = decode(col_name, u)
            out.append(decoded)
        except Exception:
            out.append(u)
    # preserve order and remove duplicates
    seen = set()
    final = []
    for x in out:
        if x not in seen:
            final.append(x)
            seen.add(x)
    return final

# ================== KPI SUMMARY =================
kpi_path = "kpi_summary.json"
if os.path.exists(kpi_path):
    try:
        with open(kpi_path, "r") as f:
            kpi = json.load(f)
    except Exception:
        kpi = {}
else:
    kpi = {}

# sensible defaults if missing
kpi_defaults = {
    "Total_Employees": len(df),
    "Attrition_Rate": kpi.get("Attrition_Rate", "N/A"),
    "Avg_Salary": kpi.get("Avg_Salary", "N/A"),
    "Avg_Tenure": kpi.get("Avg_Tenure", "N/A"),
    "Overtime_Rate": kpi.get("Overtime_Rate", "N/A"),
    "Avg_Job_Satisfaction": kpi.get("Avg_Job_Satisfaction", "N/A")
}
for k, v in kpi_defaults.items():
    if k not in kpi:
        kpi[k] = v

# ================== KPI CARDS ===================
cols = st.columns(6)
keys = list(kpi.keys())
labels = ["Total Employees", "Attrition Rate", "Avg Monthly Salary", "Avg Tenure (Years)", "Overtime Rate", "Avg Job Sat"]

for i, c in enumerate(cols):
    val = kpi.get(keys[i], "N/A") if i < len(keys) else "N/A"
    c.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{val}</div>
        <div class="kpi-label">{labels[i]}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ================== TABS ========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Attrition", "Salary", "Performance", "Clustering", "D&I", "Predictions"
])

# ------------------ TAB 1: Attrition ------------------
with tab1:
    st.header("Employee Attrition Risk")

    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 18, 60, 35)
        distance = st.slider("Distance From Home (km)", 1, 30, 10)
        income = st.number_input("Monthly Income", 1000, 20000, 5000)
        overtime = st.selectbox("Over Time", ["Yes", "No"])
    with c2:
        job_sat = st.slider("Job Satisfaction", 1, 4, 3)
        env_sat = st.slider("Environment Satisfaction", 1, 4, 3)
        travel_options = decoded_unique_options('Business Travel') or ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
        travel = st.selectbox("Business Travel", travel_options)
        role_options = decoded_unique_options('Job Role') or ["Sales Executive", "Research Scientist", "Laboratory Technician"]
        role = st.selectbox("Job Role", role_options)

    ot_val = 1 if overtime == "Yes" else 0
    travel_val = safe_transform_single('Business Travel', travel) if 'Business Travel' in le_dict else 0
    role_val = safe_transform_single('Job Role', role) if 'Job Role' in le_dict else 0

    # Build input array in the expected order (keep same features as training)
    input_data = np.array([[age, job_sat, ot_val, travel_val, income, distance, role_val, env_sat]], dtype=float)

    # Check for scaler and model availability
    if scalers.get('attrition') is None or models.get('attrition') is None:
        st.warning("Attrition model or scaler not found — cannot produce predictions here.")
    else:
        try:
            input_scaled = scalers['attrition'].transform(input_data)
            prob = models['attrition'].predict_proba(input_scaled)[0][1]
            pred = models['attrition'].predict(input_scaled)[0]
            risk_level = "High Risk" if prob > 0.3 else "Low Risk"
            st.metric("Attrition Probability", f"{prob:.1%}", delta=risk_level)
        except Exception as e:
            st.warning(f"Attrition prediction failed: {e}")

        # SHAP explanation (best-effort)
        try:
            # Build background from a subset to speed up
            feat_cols = ['Age','Job Satisfaction','Over Time','Business Travel','Monthly Income','Distance From Home','Job Role','Environment Satisfaction']
            Xbg = df[[
                'Age','Job Satisfaction','Over Time','Business Travel',
                'Monthly Income','Distance From Home','Job Role','Environment Satisfaction'
            ]].copy()

            # Convert categorical columns to numeric using safe_encode for SHAP background
            for c in ['Business Travel','Job Role']:
                if c in Xbg.columns:
                    Xbg[c] = safe_encode(c, Xbg[c])

            bg_vals = scalers['attrition'].transform(Xbg.values)
            explainer = shap.Explainer(models['attrition'], bg_vals)
            shap_out = explainer(input_scaled)
            plt.clf()
            shap.plots.waterfall(shap_out[0], show=False)
            st.pyplot(plt.gcf())
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")

        # Feature importance (if linear model)
        try:
            if hasattr(models['attrition'], 'coef_'):
                coeffs = pd.DataFrame({
                    'Feature': ['Age','Job Satisfaction','Over Time','Business Travel','Monthly Income',
                                'Distance From Home','Job Role','Environment Satisfaction'],
                    'Importance': np.abs(models['attrition'].coef_[0])
                }).sort_values('Importance', ascending=False)
                fig = px.bar(coeffs, x='Importance', y='Feature', orientation='h', title="Attrition Drivers")
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

# ------------------ TAB 2: Salary ------------------
with tab2:
    st.header("Salary Prediction")

    c1, c2 = st.columns(2)
    with c1:
        edu = st.selectbox("Education", [1,2,3,4,5],
                           format_func=lambda x: ["High School","Associates","Bachelors","Masters","PhD"][x-1])
        level = st.slider("Job Level", 1, 5, 2)
        exp = st.slider("Total Working Years", 0, 40, 10)
    with c2:
        perf = st.slider("Performance Rating", 3, 4, 3)
        role_s_options = decoded_unique_options('Job Role') or ["Sales Executive", "Research Scientist"]
        role_s = st.selectbox("Job Role", role_s_options, key="sal_role")
        stock = st.slider("Stock Option Level", 0, 3, 1)

    role_val = safe_transform_single('Job Role', role_s) if 'Job Role' in le_dict else 0
    # Construct input array consistent with training expectation
    input_sal = np.array([[edu, level, exp, perf, role_val, exp * 0.6, stock]], dtype=float)

    if scalers.get('salary') is None or models.get('salary') is None:
        st.warning("Salary model or scaler not found.")
    else:
        try:
            input_sal_s = scalers['salary'].transform(input_sal)
            pred_sal = models['salary'].predict(input_sal_s)[0]
            st.metric("Predicted Monthly Income", f"${pred_sal:,.0f}")
        except Exception as e:
            st.warning(f"Salary prediction failed: {e}")

# ------------------ TAB 3: Performance ------------------
with tab3:
    st.header("Performance & Training Impact")

    years = st.slider("Years at Company", 0, 40, 5)
    training = st.slider("Training Times Last Year", 0, 6, 3)
    involvement = st.slider("Job Involvement", 1, 4, 3)

    if models.get('performance') is None:
        st.warning("Performance model not found.")
    else:
        try:
            pred_perf = models['performance'].predict([[years, training, involvement, 2, years*1.2]])[0]
            st.metric("Predicted Performance Rating", f"{pred_perf:.2f}")
        except Exception as e:
            st.warning(f"Performance prediction failed: {e}")

# ------------------ TAB 4: Clustering ------------------
with tab4:
    st.header("Employee Segmentation")

    cluster_cols = ['CF_age band','Gender','Marital Status','Job Role','Department','Monthly Income','Job Level']
    for c in cluster_cols:
        if c not in df.columns:
            st.warning(f"Clustering column missing from dataset: {c}")
    X_cluster = df[cluster_cols].copy()

    # --- SAFE ENCODING for categorical cluster columns ---
    for col in ['CF_age band','Gender','Marital Status','Job Role','Department']:
        try:
            X_cluster[col] = safe_encode(col, X_cluster[col])
        except Exception as e:
            st.warning(f"Encoding failed for {col}: {e}")
            X_cluster[col], _ = pd.factorize(X_cluster[col].astype(str))

    # Numeric safety
    for c in ['Monthly Income','Job Level']:
        if c in X_cluster.columns:
            X_cluster[c] = pd.to_numeric(X_cluster[c], errors='coerce').fillna(0)

    # --- SCALING + KMEANS PREDICTION ---
    if scalers.get('cluster') is not None and models.get('kmeans') is not None:
        try:
            X_cluster_s = scalers['cluster'].transform(X_cluster)
            df['Cluster_Pred'] = models['kmeans'].predict(X_cluster_s)
        except Exception as e:
            st.error(f"Clustering prediction failed: {e}")
            df['Cluster_Pred'] = -1
    else:
        st.warning("Clustering model or scaler not found.")
        df['Cluster_Pred'] = -1

    # Visualization (if Age & Monthly Income available)
    if 'Age' in df.columns and 'Monthly Income' in df.columns:
        fig = px.scatter(df, x='Age', y='Monthly Income', color=df['Cluster_Pred'].astype(str),
                         size='Job Level' if 'Job Level' in df.columns else None,
                         hover_data=['Job Role'] if 'Job Role' in df.columns else None,
                         title="Employee Clusters")
        st.plotly_chart(fig, use_container_width=True)

    # Cluster summary table
    if 'Cluster_Pred' in df.columns:
        try:
            cluster_profile = df.groupby('Cluster_Pred')[['Age','Monthly Income','Job Satisfaction']].mean().round(2)
            st.dataframe(cluster_profile.style.background_gradient(cmap='Blues'))
        except Exception as e:
            st.info(f"Could not produce cluster profile: {e}")

# ------------------ TAB 5: D&I ------------------
with tab5:
    st.header("Diversity & Inclusion Insights")

    c1, c2 = st.columns(2)
    with c1:
        # decode department and gender for display
        try:
            dept_decoded = df['Department'].map(lambda x: decode('Department', x))
            gender_decoded = df['Gender'].map(lambda x: decode('Gender', x))
            gender_dept = pd.crosstab(dept_decoded, gender_decoded)
            fig = px.bar(gender_dept, barmode='group', title="Gender by Department")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"D&I chart failed: {e}")

    with c2:
        try:
            edu_decoded = df['Education Field'].map(lambda x: decode('Education Field', x))
            gender_decoded = df['Gender'].map(lambda x: decode('Gender', x))
            edu_gender = pd.crosstab(edu_decoded, gender_decoded)
            fig = px.bar(edu_gender, barmode='group', title="Gender by Education Field")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"D&I chart failed: {e}")

# ------------------ TAB 6: Predictions ------------------
with tab6:
    st.header("At-Risk & High-Potential Employees")

    # High Attrition Risk (best-effort)
    try:
        X_all = df[['Age','Job Satisfaction','Over Time','Business Travel',
                    'Monthly Income','Distance From Home','Job Role','Environment Satisfaction']].copy()

        # Map Over Time to numeric where possible
        X_all['Over Time'] = X_all['Over Time'].map({'Yes':1,'No':0}).fillna(0)

        # Convert Business Travel & Job Role to safe-encoded numeric
        for c in ['Business Travel','Job Role']:
            if c in X_all.columns:
                X_all[c] = safe_encode(c, X_all[c])

        if scalers.get('attrition') is not None and models.get('attrition') is not None:
            X_all_s = scalers['attrition'].transform(X_all.values)
            probs = models['attrition'].predict_proba(X_all_s)[:,1]
            risk_df = df.copy()
            risk_df['Risk_Prob'] = probs
            high_risk = risk_df[risk_df['Risk_Prob'] > 0.3][['Employee Number','Age','Job Role','Risk_Prob']]
            # decode Job Role for display
            high_risk['Job Role'] = high_risk['Job Role'].map(lambda x: decode('Job Role', x))
            st.subheader("High Attrition Risk (>30%)")
            st.dataframe(high_risk.sort_values('Risk_Prob', ascending=False).head(15), use_container_width=True)
        else:
            st.warning("Attrition model or scaler not available — cannot compute risk list.")
    except Exception as e:
        st.info(f"Could not compute attrition risk list: {e}")

    # Promotion Ready (best-effort)
    try:
        X_promo = df[['Performance Rating','Job Level','Years In Current Role','Total Working Years','Training Times Last Year']].copy()
        # numeric safety
        X_promo = X_promo.apply(pd.to_numeric, errors='coerce').fillna(0)

        if scalers.get('promo') is not None and models.get('promo') is not None:
            X_promo_s = scalers['promo'].transform(X_promo.values)
            promo_prob = models['promo'].predict_proba(X_promo_s)[:,1]
            promo_df = df.copy()
            promo_df['Promo_Prob'] = promo_prob
            promo_candidates = promo_df[promo_df.get('Recent_Promotion', 0) == 0].sort_values('Promo_Prob', ascending=False).head(10)
            st.subheader("Promotion Ready (Top Candidates)")
            cols_to_show = [c for c in ['Employee Number','Job Role','Performance Rating','Promo_Prob'] if c in promo_candidates.columns]
            if 'Job Role' in promo_candidates.columns:
                promo_candidates['Job Role'] = promo_candidates['Job Role'].map(lambda x: decode('Job Role', x))
            st.dataframe(promo_candidates[cols_to_show], use_container_width=True)
        else:
            st.warning("Promotion model or scaler not available — cannot compute promotion readiness.")
    except Exception as e:
        st.info(f"Promotion readiness computation failed: {e}")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("**HR Analytics Pro** • Robust label handling • Safe clustering • SHAP (best-effort) • Built with Streamlit & Plotly • Created by Godwill Okwuchukwu")
