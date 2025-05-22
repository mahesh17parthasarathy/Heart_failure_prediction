import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# ----------------------- LOGIN SECTION ----------------------- #
users = {
    "admin": "admin123",
    "user": "password"
}

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

def login():
    st.title("🔐 Login Page")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        if submit:
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.toast("✅ Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# ----------------------- MAIN APP ----------------------- #
if not st.session_state.logged_in:
    login()
else:
    st.set_page_config(page_title="Heart Failure Prediction", layout="wide")
    st.markdown(
        """
        <style>
            .stApp {
                background-image: url('image.png');
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("💓 Heart Failure Prediction System")

    selected = st.sidebar.radio("Navigate", ["🏠 Home", "📈 Predict", "📊 Evaluation", "📁 Records", "🔓 Logout"])
    st.sidebar.divider()
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")

    # Load data and preprocess
    @st.cache_data
    def load_data():
        data = pd.read_csv('heart.csv')
        df = data.copy()
        cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        le_map = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))

        age_scaler = StandardScaler()
        rbp_scaler = StandardScaler()
        chol_scaler = StandardScaler()
        maxhr_scaler = StandardScaler()
        oldpeak_scaler = MinMaxScaler()

        df['Age'] = age_scaler.fit_transform(df[['Age']])
        df['RestingBP'] = rbp_scaler.fit_transform(df[['RestingBP']])
        df['Cholesterol'] = chol_scaler.fit_transform(df[['Cholesterol']])
        df['MaxHR'] = maxhr_scaler.fit_transform(df[['MaxHR']])
        df['Oldpeak'] = oldpeak_scaler.fit_transform(df[['Oldpeak']])

        X = df.drop(columns='HeartDisease')
        y = df['HeartDisease']
        return X, y, le_map, age_scaler, rbp_scaler, chol_scaler, maxhr_scaler, oldpeak_scaler

    X, y, le_map, age_scaler, rbp_scaler, chol_scaler, maxhr_scaler, oldpeak_scaler = load_data()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    models = {
        "Logistic Regression": LogisticRegression(random_state=0, C=10),
        "SVC": SVC(kernel='linear', C=0.1, probability=True),
        "Decision Tree": DecisionTreeClassifier(random_state=1000, max_depth=4),
        "Random Forest": RandomForestClassifier(max_depth=4, random_state=0),
        "KNN": KNeighborsClassifier(leaf_size=1, n_neighbors=3, p=1)
    }

    if selected == "🏠 Home":
        st.header("Welcome 👋")
        st.write("""
            This project aims to predict the likelihood of heart failure in patients using machine learning (ML) techniques.
            By analyzing key clinical and demographic features, the system can assist healthcare professionals in early detection and intervention,
            potentially saving lives and optimizing treatment plans.
        """)
        st.image("image.png", use_container_width=True)
        st.markdown("---")

    elif selected == "📈 Predict":
        st.header("📈 Heart Disease Prediction")
        st.markdown("---")

        st.subheader("🧑 Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Full Name")
            age = st.number_input("Age", 0, 120, 50)
        with col2:
            mobile = st.text_input("Mobile No.")
            email = st.text_input("Email")

        st.divider()

        st.subheader("📋 Medical Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            sex = st.selectbox("Sex", list(le_map['Sex'].keys()))
            cp = st.selectbox("Chest Pain Type", list(le_map['ChestPainType'].keys()))
            fbs = st.selectbox("FastingBS", [0, 1])
        with col2:
            rbp = st.number_input("Resting BP", 50, 200)
            chol = st.number_input("Cholesterol", 100, 600)
            maxhr = st.number_input("MaxHR", 60, 220)
        with col3:
            ecg = st.selectbox("Resting ECG", list(le_map['RestingECG'].keys()))
            exang = st.selectbox("Exercise Angina", list(le_map['ExerciseAngina'].keys()))
            oldpeak = st.number_input("Oldpeak", 0.0, 10.0, step=0.1)
            slope = st.selectbox("ST Slope", list(le_map['ST_Slope'].keys()))

        model_choice = st.selectbox("Choose Model", list(models.keys()))

        if st.button("🔍 Predict"):
            with st.spinner("Predicting..."):
                input_data = {
                    'Age': age_scaler.transform([[age]])[0][0],
                    'Sex': le_map['Sex'][sex],
                    'ChestPainType': le_map['ChestPainType'][cp],
                    'RestingBP': rbp_scaler.transform([[rbp]])[0][0],
                    'Cholesterol': chol_scaler.transform([[chol]])[0][0],
                    'FastingBS': fbs,
                    'RestingECG': le_map['RestingECG'][ecg],
                    'MaxHR': maxhr_scaler.transform([[maxhr]])[0][0],
                    'ExerciseAngina': le_map['ExerciseAngina'][exang],
                    'Oldpeak': oldpeak_scaler.transform([[oldpeak]])[0][0],
                    'ST_Slope': le_map['ST_Slope'][slope]
                }

                input_df = pd.DataFrame([[input_data[feature] for feature in X.columns]], columns=X.columns)

                clf = models[model_choice]
                clf.fit(x_train, y_train)
                pred = clf.predict(input_df)[0]
                prob = clf.predict_proba(input_df)[0][1] if hasattr(clf, "predict_proba") else None

                st.success(f"Prediction: {'Heart Disease' if pred else 'No Heart Disease'}")
                st.metric("Probability", f"{prob:.2%}" if prob else "N/A")
                st.toast("✅ Prediction complete")

                record = pd.DataFrame([{
                    "Name": patient_name, "Age": age, "Mobile": mobile, "Email": email,
                    "Prediction": pred, "Probability": prob,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }])
                record.to_csv("patient_credentials.csv", mode='a', header=not os.path.exists("patient_credentials.csv"), index=False)
                st.success("📁 Patient record saved.")

    elif selected == "📊 Evaluation":
        st.header("📊 Model Evaluation")
        st.markdown("---")

        with st.spinner("Evaluating all models..."):
            for model_name, model in models.items():
                st.subheader(f"🔍 {model_name}")
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                # ROC Curve
                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(x_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_auc = auc(fpr, tpr)

                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, color='#2a9d8f', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='#e76f51', lw=2, linestyle='--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'ROC Curve - {model_name}')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                else:
                    st.info("⚠️ ROC Curve not available for this model.")

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
                ax.set_title(f"Confusion Matrix - {model_name}")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                # Classification Report
                st.subheader("📄 Classification Report")
                report = classification_report(y_test, y_pred)
                st.text(report)

            st.toast("✅ All model evaluations complete.")

    elif selected == "📁 Records":
        st.header("📁 Saved Patient Records")
        st.markdown("---")
        with st.spinner("Loading patient records..."):
            if os.path.exists("patient_credentials.csv"):
                df_records = pd.read_csv("patient_credentials.csv", on_bad_lines='skip')
                st.dataframe(df_records)
                st.toast("📄 Records loaded.")
            else:
                st.warning("No patient records found.")

    elif selected == "🔓 Logout":
        logout()
