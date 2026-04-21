import joblib
import pandas as pd

MODEL_PATH = "salary_predict_model.pkl"

# Required assignment input payload
input_payload = {
    "age": 7,
    "gender": 0,
    "country": 55,
    "highest_deg": 3,
    "coding_exp": 4,
    "title": 13,
    "company_size": 2,
}

# Model was trained with column names code_experience/current_title.
input_data = pd.DataFrame([
    {
        "age": input_payload["age"],
        "gender": input_payload["gender"],
        "country": input_payload["country"],
        "highest_deg": input_payload["highest_deg"],
        "code_experience": input_payload["coding_exp"],
        "current_title": input_payload["title"],
        "company_size": input_payload["company_size"],
    }
])

model = joblib.load(MODEL_PATH)
prediction = model.predict(input_data)

print(f"Predicted salary: {prediction[0]:.12f}")
