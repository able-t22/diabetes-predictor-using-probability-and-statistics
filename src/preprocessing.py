import numpy as np

def preprocess(data):
    # Replace impossible 0 values with NaN
    columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in columns:
        data[col] = data[col].replace(0, np.nan)
        data[col].fillna(data[col].mean(), inplace=True)

    return data