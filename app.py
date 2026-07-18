import sys
import os

# Add src folder to Python path
sys.path.append("src")

# Import all modules
from data_loader import load_dataset
from preprocessing import preprocess
from statistics_module import calculate_statistics
from bayes import BayesianPredictor
from visualization import (
    plot_histograms,
    correlation_heatmap,
    boxplots,
    probability_chart
)
from report import generate_report
from history import save_prediction
# Load dataset
data = load_dataset("data/diabetes.csv")

# Preprocess dataset
data = preprocess(data)

# Calculate statistics
stats = calculate_statistics(data)

print(stats)

# Show graphs
plot_histograms(data)
correlation_heatmap(data)

boxplots(data)

# Create Bayesian predictor
predictor = BayesianPredictor(data)

# Get patient details
sample = {
    "Pregnancies": float(input("Pregnancies: ")),
    "Glucose": float(input("Glucose: ")),
    "BloodPressure": float(input("Blood Pressure: ")),
    "SkinThickness": float(input("Skin Thickness: ")),
    "Insulin": float(input("Insulin: ")),
    "BMI": float(input("BMI: ")),
    "DiabetesPedigreeFunction": float(input("DPF: ")),
    "Age": float(input("Age: "))
}

# Predict
p1, p0 = predictor.predict(sample)

# Display result
generate_report(p1, p0)
probability_chart(p1,p0)
save_prediction(sample,p1,p0)