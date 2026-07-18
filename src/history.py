import os
import csv

def save_prediction(sample, diabetic_prob, normal_prob):

    file = "outputs/prediction_history.csv"

    os.makedirs("outputs", exist_ok=True)

    file_exists = os.path.isfile(file)

    with open(file, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DPF",
                "Age",
                "Diabetes %",
                "Non-Diabetes %"
            ])

        writer.writerow([
            sample["Pregnancies"],
            sample["Glucose"],
            sample["BloodPressure"],
            sample["SkinThickness"],
            sample["Insulin"],
            sample["BMI"],
            sample["DiabetesPedigreeFunction"],
            sample["Age"],
            round(diabetic_prob*100,2),
            round(normal_prob*100,2)
        ])