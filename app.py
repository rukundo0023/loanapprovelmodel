import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and encoders
model = joblib.load('loan_model.pkl')
le_job = joblib.load('jobtype_encoder.pkl')
le_marital = joblib.load('marital_encoder.pkl')
le_edu = joblib.load('education_encoder.pkl')

st.set_page_config(page_title="Loan Approval App", page_icon="🏦", layout="centered")

# --- App Title and Description ---
st.markdown("""
    <h1 style='text-align: center;'>🏦 Loan Approval App</h1>
    <p style='text-align: center;'>Check your eligibility for a $10,000 loan instantly!</p>
""", unsafe_allow_html=True)

st.image("https://cdn-icons-png.flaticon.com/512/3242/3242257.png", width=100)

st.info("This app uses a machine learning model trained on synthetic data for demonstration purposes only.")

# --- Sidebar for Inputs ---
st.sidebar.header("Applicant Details")
age = st.sidebar.slider("Age", 18, 65, 30, help="Applicant's age in years.")
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=50000, step=1000,
                                  help="Total yearly income before taxes. Cannot be negative.")
job_type = st.sidebar.selectbox("Job Type", le_job.classes_, help="Current employment status.")
credit_score = st.sidebar.slider("Credit Score", 0.3, 1.0, 0.7, help="A value between 0.3 (low) and 1.0 (high).")
marital_status = st.sidebar.selectbox("Marital Status", le_marital.classes_, help="Current marital status.")
education_level = st.sidebar.selectbox("Level of Education", le_edu.classes_, help="Highest education level.")

# --- Prepare input ---
data = {
    'Age': age,
    'Annual Income ($)': income,
    'Job Type': job_type,
    'Credit Score': credit_score,
    'Marital Status': marital_status,
    'Level of Education': education_level
}
input_df = pd.DataFrame([data])
model_input = pd.DataFrame({
    'age': [age],
    'income': [income],
    'job_type_encoded': [le_job.transform([job_type])[0]],
    'credit_score': [credit_score],
    'marital_status_encoded': [le_marital.transform([marital_status])[0]],
    'education_level_encoded': [le_edu.transform([education_level])[0]]
})

# --- Predict ---
model_features = [
    'age', 'income', 'job_type_encoded', 'credit_score',
    'marital_status_encoded', 'education_level_encoded'
]
model_input_for_pred = model_input[model_features]
approval = model.predict(model_input_for_pred)[0]
confidence = model.predict_proba(model_input_for_pred)[0][1]  # Probability of approval
risk = 1 - confidence
risk_level = 'Low' if risk < 0.4 else 'Moderate' if risk < 0.7 else 'High'
risk_color = '#2ecc40' if risk < 0.4 else '#f1c40f' if risk < 0.7 else '#e74c3c'

# --- Session State Logging ---
if 'history' not in st.session_state:
    st.session_state.history = []
st.session_state.history.append({**data, 'Approved': approval, 'Confidence': f"{confidence:.2%}"})

# --- Main Output ---
col1, col2 = st.columns([1, 2])
with col1:
    if approval:
        st.markdown(f"<h2 style='color:#27ae60;'>✅ Approved</h2>", unsafe_allow_html=True)
        st.success("You are eligible for a $10,000 loan!")
    else:
        st.markdown(f"<h2 style='color:#e74c3c;'>❌ Denied</h2>", unsafe_allow_html=True)
        st.error("Application denied: Low credit score or other risk factors.")

with col2:
    st.markdown(f"""
    <div style='padding:1em; border-radius:10px; background:{risk_color}; color:white; text-align:center;'>
        <b>Confidence:</b> {confidence:.2%} <br/>
        <b>Risk Level:</b> {risk_level}
    </div>
    """, unsafe_allow_html=True)

# --- Show input summary ---
st.subheader("📄 Your Application Summary")
st.table(input_df)

# --- Visualization (Dummy Stats) ---
st.subheader("📊 Approval History (Dummy Data)")
data_summary = pd.DataFrame(st.session_state.history)
labels = ['Approved', 'Denied']
counts = [data_summary['Approved'].sum(), len(data_summary) - data_summary['Approved'].sum()]
fig, ax = plt.subplots()
ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#27ae60', '#e74c3c'])
st.pyplot(fig)

# --- Model Info ---
with st.expander("ℹ️ About this app"):
    st.write("""
    - This demo uses a Random Forest Classifier trained on synthetic data.
    - Inputs: Age, Annual Income, Job Type, Credit Score, Marital Status, Level of Education.
    - Output: Approval status, confidence, and risk score.
    - For demonstration only; not financial advice.
    """)
