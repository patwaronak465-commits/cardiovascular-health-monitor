import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import os
import heartpy as hp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# --- Helper functions and AI model loading ---
PATIENTS_DB = "patients.csv"
REPORTS_DB = "reports.csv"
MODEL_PATH = "risk_model.joblib"
SCALER_PATH = "scaler.joblib"
REPORTS_FOLDER = "reports" # New folder for HTML reports

def create_reports_folder():
    """Ensures the reports directory exists."""
    if not os.path.exists(REPORTS_FOLDER):
        os.makedirs(REPORTS_FOLDER)

# Add your existing helper functions below
def load_patients_df():
    # ... your load_patients_df function ...
    if os.path.exists(PATIENTS_DB):
        return pd.read_csv(PATIENTS_DB)
    else:
        return pd.DataFrame(columns=[
            "patient_id", "name", "phone", "address", "age", "gender",
            "systolic_bp_untreated", "diastolic_bp_untreated",
            "systolic_bp_medicated", "diastolic_bp_medicated",
            "total_cholesterol", "hdl_cholesterol", "triglycerides",
            "smoker", "diabetic", "diabetes_medication", "family_history_cvd"
        ])

def save_patients_df(df):
    df.to_csv(PATIENTS_DB, index=False)

def load_reports_df():
    if os.path.exists(REPORTS_DB):
        return pd.read_csv(REPORTS_DB)
    else:
        return pd.DataFrame(columns=[
            "report_id", "patient_id", "analysis_date", "risk_score", "status",
            "heart_rate", "p_wave_duration", "qrs_duration", "qt_interval", "pcg_result", "report_path"
        ]) # Added report_path

def save_reports_df(df):
    df.to_csv(REPORTS_DB, index=False)

def train_and_save_model():
    data = []
    for _ in range(5000):
        age = np.random.randint(30, 90)
        gender = np.random.choice([0, 1])
        smoker = np.random.choice([True, False])
        diabetic = np.random.choice([True, False])
        diabetes_medication = np.random.choice([True, False]) if diabetic else False
        family_hist = np.random.choice([True, False])
        sbp = np.random.randint(100, 200)
        dbp = np.random.randint(60, 120)
        chol = np.random.randint(150, 300)
        hdl = np.random.randint(20, 80)
        triglycerides = np.random.randint(50, 400)
        bpm = np.random.randint(50, 120)
        sdnn = np.random.randint(20, 150)
        qrs = np.random.uniform(0.08, 0.12)
        qt = np.random.uniform(0.35, 0.45)
        p_wave = np.random.uniform(0.06, 0.11)
        risk_score = 0.0
        if age > 60: risk_score += 0.20
        if sbp > 140: risk_score += 0.15
        if chol > 220: risk_score += 0.15
        if hdl < 40: risk_score += 0.10
        if triglycerides > 200: risk_score += 0.10
        if smoker: risk_score += 0.15
        if diabetic: risk_score += 0.15
        if family_hist: risk_score += 0.10
        if qrs > 0.11: risk_score += 0.10
        if qt > 0.44: risk_score += 0.15
        status = "High Risk" if risk_score > 0.6 else "Moderate Risk" if risk_score > 0.3 else "Low Risk"
        data.append([age, gender, sbp, dbp, chol, hdl, triglycerides, smoker, diabetic, diabetes_medication, family_hist, bpm, sdnn, qrs, qt, p_wave, status])
    df_train = pd.DataFrame(data, columns=['age', 'gender', 'sbp', 'dbp', 'chol', 'hdl', 'triglycerides', 'smoker', 'diabetic', 'diabetes_medication', 'family_hist', 'bpm', 'sdnn', 'qrs', 'qt', 'p_wave', 'status'])
    features = ['age', 'gender', 'sbp', 'dbp', 'chol', 'hdl', 'triglycerides', 'smoker', 'diabetic', 'diabetes_medication', 'family_hist', 'bpm', 'sdnn', 'qrs', 'qt', 'p_wave']
    X = df_train[features]
    y = df_train['status']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_scaled, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    st.success("AI model trained and saved!")
    return model, scaler

def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    else:
        return train_and_save_model()

ai_model, scaler = load_or_train_model()

def analyze_ppg_signal(ppg_data, sample_rate):
    try:
        working_data, measures = hp.process(hp.scale_data(ppg_data), sample_rate)
        return working_data, measures
    except Exception as e:
        st.error(f"HeartPy Error: Could not process PPG signal. Details: {e}")
        return None, None

def analyze_ecg_signal(ecg_data, sample_rate):
    p_wave_duration = np.random.uniform(0.06, 0.11)
    qrs_duration = np.random.uniform(0.08, 0.12)
    qt_interval = np.random.uniform(0.35, 0.45)
    return p_wave_duration, qrs_duration, qt_interval

def analyze_pcg_signal(pcg_data):
    if np.std(pcg_data) > 0.1:
        return "Abnormal (Murmur Detected)"
    else:
        return "Normal"

def get_vascular_risk_score_ai(patient_data, signal_measures, model, scaler):
    input_features_dict = {
        'age': int(patient_data.get('age', 40)),
        'gender': 0 if patient_data.get('gender', 'Female') == 'Female' else 1,
        'sbp': int(patient_data.get('systolic_bp_untreated', patient_data.get('systolic_bp_medicated', 120))),
        'dbp': int(patient_data.get('diastolic_bp_untreated', patient_data.get('diastolic_bp_medicated', 80))),
        'chol': int(patient_data.get('total_cholesterol', 180)),
        'hdl': int(patient_data.get('hdl_cholesterol', 45)),
        'triglycerides': int(patient_data.get('triglycerides', 100)),
        'smoker': patient_data.get('smoker', False),
        'diabetic': patient_data.get('diabetic', False),
        'diabetes_medication': patient_data.get('diabetes_medication', False),
        'family_hist': patient_data.get('family_history_cvd', False),
        'bpm': signal_measures.get('bpm', 70),
        'sdnn': signal_measures.get('sdnn', 100),
        'qrs': signal_measures.get('qrs_duration', 0.1),
        'qt': signal_measures.get('qt_interval', 0.4),
        'p_wave': signal_measures.get('p_wave_duration', 0.08),
    }
    input_df = pd.DataFrame([input_features_dict])
    features = ['age', 'gender', 'sbp', 'dbp', 'chol', 'hdl', 'triglycerides', 'smoker', 'diabetic', 'diabetes_medication', 'family_hist', 'bpm', 'sdnn', 'qrs', 'qt', 'p_wave']
    input_scaled = scaler.transform(input_df[features])
    predicted_status = model.predict(input_scaled)[0]
    probas = model.predict_proba(input_scaled)
    class_labels = model.classes_
    risk_score = 0.0
    if 'High Risk' in class_labels:
        risk_score = probas[0, np.where(class_labels == 'High Risk')[0][0]]
    return min(max(risk_score, 0), 1), predicted_status

def save_report_html(report_id, df_plot, analysis):
    """Saves the full analysis report as a self-contained HTML file."""
    create_reports_folder()
    report_path = os.path.join(REPORTS_FOLDER, f"report_{report_id}.html")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head><title>Patient Report {report_id}</title></head>
    <body>
    <h1>Patient Report</h1>
    <h3>AI-Powered Risk Assessment</h3>
    <p><strong>Status:</strong> {status}</p>
    <p><strong>Estimated Blockage Risk:</strong> {risk_score:.0%}</p>
    <p><strong>Heart Rate (BPM):</strong> {heart_rate:.1f}</p>
    <p><strong>QRS Duration (s):</strong> {qrs_duration:.3f}</p>
    <p><strong>QT Interval (s):</strong> {qt_interval:.3f}</p>
    <p><strong>PCG Analysis:</strong> {pcg_result}</p>
    <h2>Multi-Sensor Waveforms</h2>
    """.format(
        report_id=report_id,
        status=analysis.get('status'),
        risk_score=analysis.get('risk_score'),
        heart_rate=analysis.get('heart_rate'),
        qrs_duration=analysis.get('qrs_duration'),
        qt_interval=analysis.get('qt_interval'),
        pcg_result=analysis.get('pcg_result')
    )

    fig_ecg = px.line(df_plot, x='timestamp', y='ecg_signal', title='ECG Waveform')
    html_content += fig_ecg.to_html(full_html=False, include_plotlyjs='cdn')
    
    fig_ppg = px.line(df_plot, x='timestamp', y='ppg_signal', title='PPG Waveform')
    html_content += fig_ppg.to_html(full_html=False, include_plotlyjs='cdn')
    
    fig_pcg = px.line(df_plot, x='timestamp', y='pcg_signal', title='PCG Waveform (Heart Sounds)')
    html_content += fig_pcg.to_html(full_html=False, include_plotlyjs='cdn')
    
    html_content += "</body></html>"
    
    with open(report_path, "w") as f:
        f.write(html_content)
        
    return report_path


# --- Login and Main App Logic ---
if 'password_correct' not in st.session_state or not st.session_state['password_correct']:
    st.set_page_config(page_title="Login", page_icon="🔐")
    st.title("🔐 Login")
    password = st.text_input("Enter password", type="password")

    if st.button("Login"):
        if password == "12345":
            st.session_state['password_correct'] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# --- Main Dashboard Logic (This part only runs if the password is correct) ---
st.set_page_config(page_title="", page_icon="📊", layout="wide")
st.title("🩺 Dashboard")

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
.main {
    background-color: #0d1117;
    color: #c9d1d9;
}
.st-emotion-cache-18ni29i.ezrtsby2 {
    visibility: hidden;
}
h1, h2, h3, h4, h5, h6 {
    color: #58a6ff;
}
.st-emotion-cache-1629p8f {
    background-color: #21262d;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}
.st-emotion-cache-1q1g0hv.e1tzp5w0 {
    background-color: #21262d;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}
.st-emotion-cache-13k65z5 {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}
.st-emotion-cache-13ln4j9 {
    background-color: #0d1117;
}
.st-emotion-cache-18ni29i {
    background-color: #0d1117;
}
.st-emotion-cache-1d90069 {
    font-size: 1.5rem;
}
.stMetric {
    background-color: #21262d;
    border-radius: 10px;
    padding: 15px;
    margin: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    color: #c9d1d9;
}
.stMetricLabel {
    color: #c9d1d9;
}
.stMetricValue {
    color: #58a6ff;
    font-size: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Add a logout button to the sidebar
with st.sidebar:
    st.button("Logout", on_click=lambda: st.session_state.update(password_correct=False))

# --- Initialize session state ---
if 'patient_loaded' not in st.session_state:
    st.session_state['patient_loaded'] = False
if 'theme' not in st.session_state:
    st.session_state['theme'] = "Dark"
if 'last_analysis' not in st.session_state:
    st.session_state['last_analysis'] = None
    
# Check if a report is being viewed
if st.session_state.get('view_report_path'):
    st.header("Saved Report")
    report_path = st.session_state.get('view_report_path')
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            html_content = f.read()
            st.components.v1.html(html_content, height=800, scrolling=True)
        st.button("⬅️ Back to Dashboard", on_click=lambda: st.session_state.pop('view_report_path'))
    else:
        st.error("Report not found.")
    st.stop()

# --- Main App Logic ---
if not st.session_state.get('patient_loaded', False):
    st.title("Patient Management")
    patients_df = load_patients_df()
    tab1, tab2 = st.tabs(["👤 Select Existing Patient", "➕ Add New Patient"])
    with tab1:
        st.header("Find and Load a Patient")
        if patients_df.empty:
            st.warning("No patients found. Please add a new patient in the next tab.")
        else:
            search_query = st.text_input("Search by Name or Patient ID", placeholder="Start typing to filter...")
            if search_query:
                mask = patients_df.apply(lambda row: str(search_query).lower() in str(row['name']).lower() or str(search_query) in str(row['patient_id']), axis=1)
                filtered_df = patients_df[mask]
            else:
                filtered_df = patients_df
            st.markdown("---")
            if not filtered_df.empty:
                with st.container(height=300):
                    for index, row in filtered_df.iterrows():
                        if st.button(f"**{row['name']}** (ID: {row['patient_id']})", key=row['patient_id'], use_container_width=True):
                            for col, val in row.items(): st.session_state[col] = val
                            st.session_state['patient_loaded'] = True
                            st.rerun()
            else:
                st.info("No matching patients found.")
    with tab2:
        st.header("Create a New Patient Record")
        with st.form("new_patient_form"):
            name = st.text_input("Full Name", placeholder="Enter full name...")
            phone = st.text_input("Phone Number", placeholder="Enter phone number...")
            address = st.text_area("Address", placeholder="Enter full address...")
            st.markdown("---")
            age = st.number_input("Age", min_value=1, max_value=120, value=None, placeholder="Enter age...")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            st.subheader("Blood Pressure")
            col_sbp, col_dbp = st.columns(2)
            with col_sbp:
                sbp_untreated = st.slider("Systolic BP (Untreated)", 90, 200, 120, step=1)
            with col_dbp:
                dbp_untreated = st.slider("Diastolic BP (Untreated)", 50, 120, 80, step=1)
            bp_meds = st.checkbox("On BP Medication?")
            if bp_meds:
                sbp_medicated = st.slider("Systolic BP (Medicated)", 90, 200, 130, step=1)
                dbp_medicated = st.slider("Diastolic BP (Medicated)", 50, 120, 85, step=1)
            else:
                sbp_medicated = None
                dbp_medicated = None
            st.subheader("Cholesterol & Lipids")
            col_chol, col_hdl = st.columns(2)
            with col_chol:
                total_cholesterol = st.slider("Total Cholesterol (mg/dL)", 150, 300, 180, step=1)
            with col_hdl:
                hdl_cholesterol = st.slider("HDL Cholesterol (mg/dL)", 20, 80, 45, step=1)
            triglycerides = st.slider("Triglycerides (mg/dL)", 50, 400, 100, step=1)
            st.subheader("Other Risk Factors")
            col_risks = st.columns(3)
            with col_risks[0]:
                smoker = st.checkbox("Smoker")
            with col_risks[1]:
                diabetic = st.checkbox("Diabetic")
            with col_risks[2]:
                diabetes_medication = st.checkbox("On Diabetes Medication?") if diabetic else False
            family_history_cvd = st.checkbox("Family History of CVD")
            if st.form_submit_button("Save and Load Patient"):
                if not name or not age:
                    st.error("Name and Age are required fields.")
                else:
                    new_id = int(time.time())
                    new_patient_data = {
                        "patient_id": new_id, "name": name, "phone": phone, "address": address,
                        "age": age, "gender": gender,
                        "systolic_bp_untreated": sbp_untreated, "diastolic_bp_untreated": dbp_untreated,
                        "systolic_bp_medicated": sbp_medicated, "diastolic_bp_medicated": dbp_medicated,
                        "total_cholesterol": total_cholesterol, "hdl_cholesterol": hdl_cholesterol, "triglycerides": triglycerides,
                        "smoker": smoker, "diabetic": diabetic, "diabetes_medication": diabetes_medication, "family_history_cvd": family_history_cvd
                    }
                    new_df = pd.DataFrame([new_patient_data])
                    updated_df = pd.concat([patients_df, new_df], ignore_index=True)
                    save_patients_df(updated_df)
                    for key, val in new_patient_data.items(): st.session_state[key] = val
                    st.session_state['patient_loaded'] = True
                    st.success(f"Patient {name} saved with ID {new_id}.")
                    time.sleep(1)
                    st.rerun()
else:
    # --- Main Dashboard ---
    name = st.session_state.get('name', 'N/A')
    patient_id = st.session_state.get('patient_id', 'N/A')
    col_head, col_btn = st.columns([0.8, 0.2])
    with col_head:
        st.markdown(f"<h1>🩺 Dashboard for: {name} <span style='font-size: 1.2rem; color: #aaa;'> (ID: {patient_id})</span></h1>", unsafe_allow_html=True)
    with col_btn:
        st.button("⬅️ Change Patient", on_click=lambda: st.session_state.update({'patient_loaded': False}), use_container_width=True)
    tab_dashboard, tab_reports = st.tabs(["📊 Live Analysis", "📜 Patient Reports"])
    with tab_dashboard:
        st.subheader("Real-time Vitals & Risk Assessment")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric(label="Age", value=st.session_state.get('age', 'N/A'))
        with metrics_col2:
            current_bp = st.session_state.get('systolic_bp_medicated', st.session_state.get('systolic_bp_untreated'))
            st.metric(label="Systolic BP", value=f"{current_bp} mmHg")
        with metrics_col3:
            current_chol = st.session_state.get('total_cholesterol', 'N/A')
            st.metric(label="Total Cholesterol", value=f"{current_chol} mg/dL")
        st.markdown("---")
        col_analysis, col_risk = st.columns([1, 2])
        with col_analysis:
            with st.container():
                st.subheader("📂 New Analysis")
                uploaded_file = st.file_uploader("Upload Signal Data CSV", type=["csv"])
                if uploaded_file:
                    if st.button("Run Analysis", use_container_width=True, help="Analyze the uploaded multi-sensor data."):
                        df = pd.read_csv(uploaded_file)
                        required_cols = ['timestamp', 'ecg_signal', 'ppg_signal', 'pcg_signal']
                        if not all(col in df.columns for col in required_cols):
                            st.error(f"CSV file must contain columns: {', '.join(required_cols)}")
                        else:
                            with st.spinner("Analyzing multi-sensor data..."):
                                ppg_signal = pd.to_numeric(df['ppg_signal'], errors='coerce').fillna(0).values
                                ecg_signal = pd.to_numeric(df['ecg_signal'], errors='coerce').fillna(0).values
                                pcg_signal = pd.to_numeric(df['pcg_signal'], errors='coerce').fillna(0).values
                                sample_rate = 100
                                working_data, ppg_measures = analyze_ppg_signal(ppg_signal, sample_rate)
                                pcg_result = analyze_pcg_signal(pcg_signal)
                                p_wave, qrs, qt = analyze_ecg_signal(ecg_signal, sample_rate)
                                if ppg_measures:
                                    patient_data_for_model = st.session_state.to_dict()
                                    signal_measures = {
                                        'bpm': ppg_measures.get('bpm', 70),
                                        'sdnn': ppg_measures.get('sdnn', 100),
                                        'qrs_duration': qrs,
                                        'qt_interval': qt,
                                        'p_wave_duration': p_wave
                                    }
                                    risk_score, status = get_vascular_risk_score_ai(patient_data_for_model, signal_measures, ai_model, scaler)
                                    bpm = ppg_measures.get('bpm', 0)
                                    st.session_state['last_analysis'] = {
                                        "risk_score": risk_score, "status": status, "heart_rate": bpm,
                                        "p_wave_duration": p_wave, "qrs_duration": qrs, "qt_interval": qt,
                                        "pcg_result": pcg_result, "df": df
                                    }
                                    report_id = int(time.time())
                                    analysis_date = time.strftime('%Y-%m-%d %H:%M:%S')
                                    new_report_data = {
                                        "report_id": report_id, "patient_id": patient_id, "analysis_date": analysis_date,
                                        "risk_score": risk_score, "status": status, "heart_rate": bpm,
                                        "p_wave_duration": p_wave, "qrs_duration": qrs, "qt_interval": qt, "pcg_result": pcg_result
                                    }
                                    new_report_df = pd.DataFrame([new_report_data])
                                    updated_reports_df = pd.concat([load_reports_df(), new_report_df], ignore_index=True)
                                    save_reports_df(updated_reports_df)
                                    st.success("Analysis complete and report saved!")
                                    st.rerun()
        with col_risk:
            if st.session_state['last_analysis']:
                analysis = st.session_state['last_analysis']
                st.subheader("AI-Powered Risk Assessment")
                risk_status_col, risk_score_col = st.columns(2)
                with risk_status_col:
                    status = analysis.get('status', 'N/A')
                    if status == "High Risk": st.error(f"**Status: {status}**")
                    elif status == "Moderate Risk": st.warning(f"**Status: {status}**")
                    elif status == "Low Risk": st.success(f"**Status: {status}**")
                    else: st.info(f"**Status: No analysis data**")
                    risk_score = analysis.get('risk_score', 0)
                    st.progress(risk_score)
                    st.metric(label="Estimated Blockage Risk", value=f"{risk_score:.0%}")
                with risk_score_col:
                    st.write("### Vitals from Analysis")
                    st.metric(label="Heart Rate (BPM)", value=f"{analysis.get('heart_rate', 0):.1f}")
                    st.metric(label="QRS Duration (s)", value=f"{analysis.get('qrs_duration', 0.0):.3f}")
                    st.metric(label="QT Interval (s)", value=f"{analysis.get('qt_interval', 0.0):.3f}")
                    st.metric(label="PCG Analysis", value=analysis.get('pcg_result', 'N/A'))
            else:
                st.info("Upload a CSV and run an analysis to see results here.")
        st.markdown("---")
        if st.session_state['last_analysis']:
            st.subheader("Multi-Sensor Waveforms")
            df_plot = st.session_state['last_analysis']['df']
            current_theme = st.session_state.get("theme", "Dark")
            plot_theme = "plotly_dark" if current_theme == "Dark" else "plotly_white"
            fig_ecg = px.line(df_plot, x='timestamp', y='ecg_signal', title='ECG Waveform')
            fig_ecg.update_layout(template=plot_theme)
            st.plotly_chart(fig_ecg, use_container_width=True)
            fig_ppg = px.line(df_plot, x='timestamp', y='ppg_signal', title='PPG Waveform')
            fig_ppg.update_layout(template=plot_theme)
            st.plotly_chart(fig_ppg, use_container_width=True)
            fig_pcg = px.line(df_plot, x='timestamp', y='pcg_signal', title='PCG Waveform (Heart Sounds)')
            fig_pcg.update_layout(template=plot_theme)
            st.plotly_chart(fig_pcg, use_container_width=True)
    with tab_reports:
        st.header("Patient Reports")
        reports_df = load_reports_df()
        patient_reports = reports_df[reports_df['patient_id'] == patient_id]
        if patient_reports.empty:
            st.info("No previous reports found for this patient.")
        else:
            st.subheader("Report History")
            st.dataframe(patient_reports.sort_values(by='analysis_date', ascending=False), use_container_width=True)
            st.subheader("Risk Score Trend Over Time")
            current_theme = st.session_state.get("theme", "Dark")
            plot_theme = "plotly_dark" if current_theme == "Dark" else "plotly_white"
            fig_trend = px.line(patient_reports, x='analysis_date', y='risk_score', title='Risk Score Trend')
            fig_trend.update_layout(template=plot_theme)
            st.plotly_chart(fig_trend, use_container_width=True)
    with st.sidebar:
        st.header("Update Vitals")
        new_sbp_untreated = st.slider("Systolic BP (Untreated)", 90, 200, int(st.session_state.get('systolic_bp_untreated', 120)), step=1)
        new_dbp_untreated = st.slider("Diastolic BP (Untreated)", 50, 120, int(st.session_state.get('diastolic_bp_untreated', 80)), step=1)
        bp_meds = st.checkbox("On BP Medication?", value=st.session_state.get('systolic_bp_medicated', False) is not None)
        if bp_meds:
            new_sbp_medicated = st.slider("Systolic BP (Medicated)", 90, 200, int(st.session_state.get('systolic_bp_medicated', 130)), step=1)
            new_dbp_medicated = st.slider("Diastolic BP (Medicated)", 50, 120, int(st.session_state.get('diastolic_bp_medicated', 85)), step=1)
        else:
            new_sbp_medicated = None
            new_dbp_medicated = None
        new_total_cholesterol = st.slider("Total Cholesterol (mg/dL)", 150, 300, int(st.session_state.get('total_cholesterol', 180)), step=1)
        new_hdl = st.slider("HDL Cholesterol (mg/dL)", 20, 80, int(st.session_state.get('hdl_cholesterol', 45)), step=1)
        new_triglycerides = st.slider("Triglycerides (mg/dL)", 50, 400, int(st.session_state.get('triglycerides', 100)), step=1)
        st.markdown("---")
        new_smoker = st.checkbox("Smoker", value=st.session_state.get('smoker', False))
        new_diabetic = st.checkbox("Diabetic", value=st.session_state.get('diabetic', False))
        new_diabetes_meds = st.checkbox("On Diabetes Medication?", value=st.session_state.get('diabetes_medication', False)) if new_diabetic else False
        new_family_history = st.checkbox("Family History of CVD", value=st.session_state.get('family_history_cvd', False))
        if st.button("Save Vital Updates"):
            patients_df = load_patients_df()
            patients_df.loc[patients_df['patient_id'] == st.session_state['patient_id'],
                             ['systolic_bp_untreated', 'diastolic_bp_untreated', 'systolic_bp_medicated', 'diastolic_bp_medicated',
                              'total_cholesterol', 'hdl_cholesterol', 'triglycerides',
                              'smoker', 'diabetic', 'diabetes_medication', 'family_history_cvd']] = [
                                  new_sbp_untreated, new_dbp_untreated, new_sbp_medicated, new_dbp_medicated,
                                  new_total_cholesterol, new_hdl, new_triglycerides,
                                  new_smoker, new_diabetic, new_diabetes_meds, new_family_history
                              ]
            save_patients_df(patients_df)
            st.session_state['systolic_bp_untreated'] = new_sbp_untreated
            st.session_state['diastolic_bp_untreated'] = new_dbp_untreated
            st.session_state['systolic_bp_medicated'] = new_sbp_medicated
            st.session_state['diastolic_bp_medicated'] = new_dbp_medicated
            st.session_state['total_cholesterol'] = new_total_cholesterol
            st.session_state['hdl_cholesterol'] = new_hdl
            st.session_state['triglycerides'] = new_triglycerides
            st.session_state['smoker'] = new_smoker
            st.session_state['diabetic'] = new_diabetic
            st.session_state['diabetes_medication'] = new_diabetes_meds
            st.session_state['family_history_cvd'] = new_family_history
            st.success("Vitals updated successfully!")
            time.sleep(1)
            st.rerun()
