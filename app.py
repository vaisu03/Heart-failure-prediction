import streamlit as st
import os
import sqlite3
import bcrypt
import joblib
import numpy as np
import pandas as pd

# =========================
# DB Setup (same as before)
conn = sqlite3.connect("hospital.db", check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS doctors(username TEXT UNIQUE, password BLOB)')
c.execute('''
    CREATE TABLE IF NOT EXISTS patients(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doctor TEXT, name TEXT,
        age INTEGER, anaemia INTEGER, creatinine_phosphokinase REAL,
        diabetes INTEGER, ejection_fraction REAL, high_blood_pressure INTEGER,
        platelets REAL, serum_creatinine REAL, serum_sodium REAL,
        sex INTEGER, smoking INTEGER, time INTEGER,
        prediction INTEGER, probability REAL
    )
''')
conn.commit()

# =========================
# DB Functions (unchanged)
def add_doctor(username, password):
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    c.execute('INSERT OR IGNORE INTO doctors(username, password) VALUES (?,?)', (username, hashed))
    conn.commit()

def login_doctor(username, password):
    c.execute('SELECT password FROM doctors WHERE username=?', (username,))
    row = c.fetchone()
    if row and bcrypt.checkpw(password.encode('utf-8'), row[0]):
        return True
    return False

def save_patient(doctor, name, features, prediction, probability):
    c.execute('''
        INSERT INTO patients(
            doctor, name, age, anaemia, creatinine_phosphokinase, diabetes,
            ejection_fraction, high_blood_pressure, platelets,
            serum_creatinine, serum_sodium, sex, smoking, time,
            prediction, probability
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (doctor, name) + tuple(features) + (int(prediction), float(probability)))
    conn.commit()

def get_patients(doctor):
    c.execute('SELECT id, name, prediction, probability FROM patients WHERE doctor=?', (doctor,))
    return c.fetchall()

def get_patient_details(patient_id):
    c.execute('SELECT * FROM patients WHERE id=?', (patient_id,))
    return c.fetchone()

# =========================
# Load Model & Feature Names
model_path = os.path.join(os.path.dirname(__file__), "heart_failure_model.pkl")
model = joblib.load(model_path)

FEATURE_COLUMNS = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time",
    "age_time_interaction", "ejection_time_ratio"
]

def process_input(age, anaemia, creatinine_phosphokinase, diabetes,
                  ejection_fraction, high_blood_pressure, platelets,
                  serum_creatinine, serum_sodium, sex, smoking, time):
    row = pd.DataFrame([[ 
        age, anaemia, np.log1p(creatinine_phosphokinase), diabetes,
        ejection_fraction, high_blood_pressure, platelets,
        np.log1p(serum_creatinine), serum_sodium, sex, smoking, time
    ]], columns=[
        "age", "anaemia", "creatinine_phosphokinase", "diabetes",
        "ejection_fraction", "high_blood_pressure", "platelets",
        "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
    ])
    row["age_time_interaction"] = row["age"] * row["time"]
    row["ejection_time_ratio"] = row["ejection_fraction"] / (row["time"] + 1)
    return row[FEATURE_COLUMNS]

# =========================
# Streamlit UI & Styling

st.set_page_config(page_title="Heart Failure Risk App", layout="wide")

# CSS for background and styling
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('https://images.unsplash.com/photo-1582931453207-0e9055a03b5c?crop=entropy&cs=tinysrgb&fit=crop&fm=jpg&h=1080&w=1920');
    background-size: cover;
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.8);
}
h1, h2, h3, .title {
    color: #2C3E50;
}
.stButton>button {
    background-color: #d9534f;
    color: white;
    border: none;
}
.stButton>button:hover {
    background-color: #c9302c;
    color: white;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "doctor" not in st.session_state:
    st.session_state.doctor = None

menu = ["Home", "Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

# Home
if choice == "Home":
    st.markdown("<h1 style='text-align:center;'>❤️ Heart Failure Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>AI-powered tool for doctors</h3>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=150)

elif choice == "Register":
    st.subheader("👨‍⚕️ Doctor Registration")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Register"):
        if new_user and new_pass:
            add_doctor(new_user, new_pass)
            st.success("Account created! Please login.")
        else:
            st.error("Please fill both fields.")

elif choice == "Login":
    if not st.session_state.logged_in:
        st.subheader("🔐 Doctor Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_doctor(username, password):
                st.session_state.logged_in = True
                st.session_state.doctor = username
                st.success(f"Welcome Dr. {username}!")
                st.rerun()
            else:
                st.error("Invalid login.")
    else:
        st.success(f"Welcome Dr. {st.session_state.doctor}! 👨‍⚕️")
        tabs = st.tabs(["📊 Predict", "📁 Dashboard"])

        with tabs[0]:
            st.header("📊 Predict Heart Failure Risk")
            name = st.text_input("Patient Name")
            age = st.number_input("Age", 20, 100, 50)
            anaemia = st.selectbox("Anaemia", [0, 1])
            cpk = st.number_input("Creatinine Phosphokinase", 1.0, 10000.0, 150.0)
            diabetes = st.selectbox("Diabetes", [0, 1])
            ef = st.slider("Ejection Fraction (%)", 10, 80, 40)
            hbp = st.selectbox("High Blood Pressure", [0, 1])
            platelets = st.number_input("Platelets", 100000.0, 600000.0, 250000.0)
            sc = st.number_input("Serum Creatinine", 0.1, 20.0, 1.0)
            ss = st.number_input("Serum Sodium", 100.0, 150.0, 135.0)
            sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
            smoking = st.selectbox("Smoking", [0, 1])
            time = st.number_input("Follow-up Time", 1, 300, 100)

            if st.button("Predict"):
                row = process_input(age, anaemia, cpk, diabetes, ef, hbp, platelets, sc, ss, sex, smoking, time)
                probability = model.predict_proba(row)[0][1]
                risk_threshold = 0.25
                prediction = int(probability >= risk_threshold)

                # Show a progress bar gauge
                st.write("## Risk Score")
                st.progress(min(1.0, probability))  # show a bar up to 100%
                
                if prediction == 1:
                    st.error(f"⚠️ **High Risk** (Probability: {probability:.2f})")
                else:
                    st.success(f"✅ **Low Risk** (Probability: {probability:.2f})")

                save_patient(
                    st.session_state.doctor,
                    name,
                    [age, anaemia, cpk, diabetes, ef, hbp, platelets, sc, ss, sex, smoking, time],
                    prediction,
                    probability
                )

        with tabs[1]:
            st.header("📁 Patients Dashboard")
            patients = get_patients(st.session_state.doctor)
            if patients:
                for pid, pname, pred, prob in patients:
                    risk = "⚠️ High Risk" if pred == 1 else "✅ Low Risk"
                    if st.button(f"{pname} ({risk}, {prob:.2f})", key=pid):
                        details = get_patient_details(pid)
                        cols = ["ID","Doctor","Name","Age","Anaemia","CPK","Diabetes",
                                "Ejection Fraction","High BP","Platelets","Serum Creatinine",
                                "Serum Sodium","Sex","Smoking","Time","Prediction","Probability"]
                        df = pd.DataFrame([details], columns=cols)
                        st.subheader(f"Details for {pname}")
                        st.table(df)
            else:
                st.info("No patients recorded yet.")
