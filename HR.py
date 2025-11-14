# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import json
import pickle
import os

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
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .kpi-label {
        font-size: 1rem;
        color: #555;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ================== TITLE ======================
st.title("HR Analytics Pro")
st.markdown("### Predictive HR Insights • Clustering • SHAP • KPI Dashboard")

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    return pd.read_csv("hr_processed.csv")

# ================== LOAD MODELS =================
@st.cache_resource
def load_models():
    models, scalers, le_dict = {}, {}, {}
    missing = []

    def safe_load(path, loader=joblib.load):
        if not os.path.exists(path):
            return None
        try:
            return loader(path)
        except Exception as e:
            st.error(f"Failed to load {path}: {e}")
            return None

    models['attrition'] = safe_load("attr_model.pkl")
    scalers['attrition'] = safe_load("scaler_attr.pkl")

    models['performance'] = safe_load("perf_model.pkl")
    models['salary'] = safe_load("sal_model.pkl")
    scalers['salary'] = safe_load("sal_scaler.pkl")

    models['kmeans'] = safe_load("kmeans_model.pkl")
    scalers['cluster'] = safe_load("cluster_scaler.pkl")

    models['promo'] = safe_load("promo_model.pkl")
    scalers['promo'] = safe_load("promo_scaler.pkl")

    models['ot'] = safe_load("ot_model.pkl")
    scalers['ot'] = safe_load("ot_scaler.pkl")

    models['tenure'] = safe_load("tenure_model.pkl")
    models['train_effect'] = safe_load("train_effect_model.pkl")

    le_path = "le_dict.pkl"
    if os.path.exists(le_path):
        try:
            with open(le_path, "rb") as f:
                le_dict = pickle.load(f)
        except Exception as e:
            st.error(f"Failed to load label encoders (le_dict.pkl): {e}")
            le_dict = {}
    else:
        st.warning("⚠️ le_dict.pkl not found — decoding may not work.")
        le_dict = {}

    for k, v in {**models, **scalers}.items():
        if v is None:
            missing.append(k)
    if missing:
        st.warning(f"⚠️ Missing or failed to load: {missing}")

    return models, scalers, le_dict


# ================== LOAD ALL ===================
df = load_data()
models, scalers, le_dict = load_models()

if models is None or scalers is None:
    st.stop()

# ================== DECODE FUNCTIONS =============
def decode(col, val):
    if col in le_dict and hasattr(le_dict[col], 'inverse_transform'):
        try:
            if isinstance(val, (str,)) and val.isdigit():
                val = int(val)
            return le_dict[col].inverse_transform([int(val)])[0]
        except Exception:
            return val
    return val

def decoded_unique_options(col_name):
    uniques = df[col_name].unique()
    out = []
    for u in uniques:
        try:
            out.append(decode(col_name, u))
        except Exception:
            out.append(u)
    # remove duplicates while preserving order
    seen, final = set(), []
    for item in out:
        if item not in seen:
            final.append(item)
            seen.add(item)
    return final

# ================== KPI SUMMARY =================
if os.path.exists("kpi_summary.json"):
    with open("kpi_summary.json", "r") as f:
        kpi = json.load(f)
else:
    kpi = {
        "Total_Employees": len(df),
        "Attrition_Rate": "N/A",
        "Avg_Salary": "N/A",
        "Avg_Tenure": "N/A",
        "Overtime_Rate": "N/A",
        "Avg_Job_Satisfaction": "N/A"
    }

# ================== KPI CARDS ===================
cols = st.columns(6)
keys = list(kpi.keys())
labels = ["Total Employees", "Attrition Rate", "Avg Monthly Salary", "Avg Tenure (Years)", "Overtime Rate", "Avg Job Sat"]

for i, c in enumerate(cols):
    c.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{kpi[keys[i]]}</div>
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
        travel = st.selectbox("Business Travel", decoded_unique_options('Business Travel'))
        role = st.selectbox("Job Role", decoded_unique_options('Job Role'))

    ot_val = 1 if overtime == "Yes" else 0
    travel_val = le_dict['Business Travel'].transform([travel])[0] if 'Business Travel' in le_dict else 0
    role_val = le_dict['Job Role'].transform([role])[0] if 'Job Role' in le_dict else 0

    input_data = np.array([[age, job_sat, ot_val, travel_val, income, distance, role_val, env_sat]])
    input_scaled = scalers['attrition'].transform(input_data)

    prob = models['attrition'].predict_proba(input_scaled)[0][1]
    pred = models['attrition'].predict(input_scaled)[0]
    risk_level = "High Risk" if prob > 0.3 else "Low Risk"

    st.metric("Attrition Probability", f"{prob:.1%}", delta=risk_level)

    # SHAP Explanation (robust)
    try:
        features = ['Age','Job Satisfaction','Over Time','Business Travel',
                    'Monthly Income','Distance From Home','Job Role','Environment Satisfaction']
        bg = scalers['attrition'].transform(df[[
            'Age','Job Satisfaction','Over Time','Business Travel',
            'Monthly Income','Distance From Home','Job Role','Environment Satisfaction'
        ]].values)

        explainer = shap.Explainer(models['attrition'], bg)
        shap_out = explainer(input_scaled)
        plt.clf()
        shap.plots.waterfall(shap_out[0], show=False)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")

    # Feature importance
    coeffs = pd.DataFrame({
        'Feature': ['Age','Job Satisfaction','Over Time','Business Travel','Monthly Income',
                    'Distance From Home','Job Role','Environment Satisfaction'],
        'Importance': np.abs(models['attrition'].coef_[0])
    }).sort_values('Importance', ascending=False)
    fig = px.bar(coeffs, x='Importance', y='Feature', orientation='h', title="Attrition Drivers")
    st.plotly_chart(fig, use_container_width=True)

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
        role_s = st.selectbox("Job Role", decoded_unique_options('Job Role'), key="sal_role")
        stock = st.slider("Stock Option Level", 0, 3, 1)

    role_val = le_dict['Job Role'].transform([role_s])[0] if 'Job Role' in le_dict else 0
    input_sal = np.array([[edu, level, exp, perf, role_val, exp*0.6, stock]])
    input_sal_s = scalers['salary'].transform(input_sal)
    pred_sal = models['salary'].predict(input_sal_s)[0]

    st.metric("Predicted Monthly Income", f"${pred_sal:,.0f}")

# ------------------ TAB 3: Performance ------------------
with tab3:
    st.header("Performance & Training Impact")

    years = st.slider("Years at Company", 0, 40, 5)
    training = st.slider("Training Times Last Year", 0, 6, 3)
    involvement = st.slider("Job Involvement", 1, 4, 3)

    pred_perf = models['performance'].predict([[years, training, involvement, 2, years*1.2]])[0]
    st.metric("Predicted Performance Rating", f"{pred_perf:.2f}")

# ------------------ TAB 4: Clustering ------------------
with tab4:
    st.header("Employee Segmentation")

    cluster_cols = ['CF_age band','Gender','Marital Status','Job Role','Department','Monthly Income','Job Level']
    X_cluster = df[cluster_cols].copy()

    for col in ['CF_age band','Gender','Marital Status','Job Role','Department']:
        if col in le_dict:
            try:
                X_cluster[col] = le_dict[col].transform(X_cluster[col].astype(str))
            except Exception as e:
                st.warning(f"Encoding failed for {col}: {e}")
        else:
            X_cluster[col], _ = pd.factorize(X_cluster[col].astype(str))

    if scalers.get('cluster') and models.get('kmeans'):
        X_cluster_s = scalers['cluster'].transform(X_cluster)
        df['Cluster_Pred'] = models['kmeans'].predict(X_cluster_s)

        fig = px.scatter(df, x='Age', y='Monthly Income', color=df['Cluster_Pred'].astype(str),
                         size='Job Level', hover_data=['Job Role'], title="Employee Clusters")
        st.plotly_chart(fig, use_container_width=True)

        cluster_profile = df.groupby('Cluster_Pred')[['Age','Monthly Income','Job Satisfaction']].mean().round(2)
        st.dataframe(cluster_profile.style.background_gradient(cmap='Blues'))
    else:
        st.warning("Clustering model or scaler not found.")

# ------------------ TAB 5: D&I ------------------
with tab5:
    st.header("Diversity & Inclusion Insights")

    c1, c2 = st.columns(2)
    with c1:
        gender_dept = pd.crosstab(df['Department'].map(lambda x: decode('Department', x)),
                                  df['Gender'].map(lambda x: decode('Gender', x)))
        fig = px.bar(gender_dept, barmode='group', title="Gender by Department")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        edu_gender = pd.crosstab(df['Education Field'].map(lambda x: decode('Education Field', x)),
                                 df['Gender'].map(lambda x: decode('Gender', x)))
        fig = px.bar(edu_gender, barmode='group', title="Gender by Education Field")
        st.plotly_chart(fig, use_container_width=True)

# ------------------ TAB 6: Predictions ------------------
with tab6:
    st.header("At-Risk & High-Potential Employees")

    # High Attrition Risk
    X_all = df[['Age','Job Satisfaction','Over Time','Business Travel',
                'Monthly Income','Distance From Home','Job Role','Environment Satisfaction']].copy()
    X_all['Over Time'] = X_all['Over Time'].map({'Yes':1,'No':0}).fillna(0)

    X_all_s = scalers['attrition'].transform(X_all)
    probs = models['attrition'].predict_proba(X_all_s)[:,1]
    risk_df = df.copy()
    risk_df['Risk_Prob'] = probs
    high_risk = risk_df[risk_df['Risk_Prob'] > 0.3][['Employee Number','Age','Job Role','Risk_Prob']]
    high_risk['Job Role'] = high_risk['Job Role'].map(lambda x: decode('Job Role', x))

    st.subheader("High Attrition Risk (>30%)")
    st.dataframe(high_risk.sort_values('Risk_Prob', ascending=False).head(15), use_container_width=True)

    # Promotion Ready
    X_promo = df[['Performance Rating','Job Level','Years In Current Role','Total Working Years','Training Times Last Year']]
    X_promo_s = scalers['promo'].transform(X_promo)
    promo_prob = models['promo'].predict_proba(X_promo_s)[:,1]
    promo_df = df[df['Recent_Promotion'] == 0].copy()
    promo_df['Promo_Prob'] = promo_prob[df['Recent_Promotion'] == 0]

    st.subheader("Promotion Ready (Top Candidates)")
    top_promo = promo_df.sort_values('Promo_Prob', ascending=False).head(10)
    st.dataframe(top_promo[['Employee Number','Job Role','Performance Rating','Promo_Prob']], use_container_width=True)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("**HR Analytics Pro** • Built with Streamlit, SHAP, and Plotly • Empowering Data-Driven HR Decisions")
