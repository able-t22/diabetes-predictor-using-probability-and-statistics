from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = ROOT / "src"
DATA_PATH = ROOT / "data" / "diabetes.csv"
sys.path.insert(0, str(SRC_PATH))
from data_loader import load_dataset
from preprocessing import preprocess
from bayes import BayesianPredictor
data = load_dataset(DATA_PATH)
data = preprocess(data)
predictor = BayesianPredictor(data)
st.set_page_config(
    page_title="Diabetes Analyzer",
    page_icon="🩺",
    layout="wide"
)
st.title("🩺 Diabetes Mellitus Prediction Analyzer")
st.markdown("### Bayesian Probability & Statistical Inference")
st.sidebar.header("Patient Information")
pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose", 0, 300, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 200, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 25)
insulin = st.sidebar.number_input("Insulin", 0, 900, 100)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.50)
age = st.sidebar.number_input("Age", 1, 120, 30)
predict_button = st.sidebar.button("🔍 Predict")
tab1, tab2, tab3 = st.tabs(
    ["Prediction", "Dataset Analysis", "Visualizations"]
)
with tab1:
    st.subheader("Prediction Dashboard")
    if predict_button:
        sample = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }
        p1, p0 = predictor.predict(sample)
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Probability of Diabetes",
                f"{p1*100:.2f}%"
            )
            st.progress(float(p1))
        with col2:
            st.metric(
                "Probability of Non-Diabetes",
                f"{p0*100:.2f}%"
            )
            st.progress(float(p0))
        if p1 >= 0.75:
            st.error("🔴 HIGH RISK OF DIABETES")
        elif p1 >= 0.50:
            st.warning("🟠 MODERATE RISK")
        else:
            st.success("🟢 LOW RISK")
        fig = px.bar(
            x=["Diabetes", "Non-Diabetes"],
            y=[p1*100, p0*100],
            color=["Diabetes", "Non-Diabetes"],
            title="Prediction Probability",
            labels={"x":"Class","y":"Probability (%)"}
        )
        st.plotly_chart(fig, use_container_width=True)
        comparison = pd.DataFrame({
            "Feature":["Glucose","BMI","Age","BloodPressure"],
            "Patient":[
                glucose,
                bmi,
                age,
                bp
            ],
            "Dataset Average":[
                data["Glucose"].mean(),
                data["BMI"].mean(),
                data["Age"].mean(),
                data["BloodPressure"].mean()
            ]
        })
        fig2 = px.bar(
            comparison,
            x="Feature",
            y=["Patient","Dataset Average"],
            barmode="group",
            title="Patient vs Dataset Average"
        )
        st.plotly_chart(fig2, use_container_width=True)
with tab2:
    st.subheader("Dataset Preview")
    st.dataframe(data.head(15))
    st.subheader("Statistical Summary")
    st.dataframe(data.describe())
with tab3:
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(
        data.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)
    st.divider()
    feature = st.selectbox(
        "Select Feature",
        data.columns[:-1]
    )
    fig = px.histogram(
        data,
        x=feature,
        color="Outcome",
        marginal="box",
        title=f"{feature} Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
