import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# --- PAGE SETUP ---
st.set_page_config(page_title="Medi-Trust AI", layout="wide")

# --- MODEL TRAINING ---
@st.cache_resource
def train_model():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(1000, 5), columns=['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin'])
    X['Glucose'] *= 200 
    X['BMI'] *= 50
    X['Age'] *= 80
    X['BloodPressure'] *= 120
    X['Insulin'] *= 300
    y = ((X['Glucose'] > 125) | (X['BMI'] > 30)).astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X

model, X_train = train_model()

# --- UI INTERFACE ---
st.title("🏥 Medi-Trust: Explainable Health AI")
st.write("Diagnostic Dashboard for Healthcare Risk Assessment")

st.sidebar.header("Patient Metrics")
glucose = st.sidebar.slider("Glucose Level", 70, 200, 110)
bmi = st.sidebar.slider("BMI Index", 15.0, 50.0, 24.0)
age = st.sidebar.slider("Age", 1, 100, 30)
bp = st.sidebar.slider("Blood Pressure", 40, 140, 80)
insulin = st.sidebar.slider("Insulin Level", 0, 800, 80)

user_input = pd.DataFrame([[glucose, bmi, age, bp, insulin]], columns=X_train.columns)

if st.button("Analyze Health Risk"):
    prob = model.predict_proba(user_input)[0][1]
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction")
        if prob > 0.5:
            st.error(f"High Risk Detected ({prob*100:.1f}%)")
        else:
            st.success(f"Low Risk / Healthy ({prob*100:.1f}%)")
        st.write("---")
        st.info("Red bars below mean that factor increased risk, Blue means it decreased risk.")
            
    with col2:
        st.subheader("💡 Why this prediction?")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(user_input)
        
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]

        fig, ax = plt.subplots()
        colors = ['red' if x > 0 else 'blue' for x in vals]
        pd.Series(vals, index=X_train.columns).plot(kind='barh', color=colors, ax=ax)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.title("Impact of Health Factors")
        st.pyplot(fig
