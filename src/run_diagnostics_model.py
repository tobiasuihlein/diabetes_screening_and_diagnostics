print("- loading program")
import sys
import os
import pandas as pd
import joblib

if __name__ == "__main__":

    # get data from user input
    print('Please enter data')
    age_ = input("Age: ")
    hypertension_ = input("Hypertension (1/0): ")
    heart_disease_ = input("Heart Disease (1/0): ")
    bmi_ = input("BMI: ")
    HbA1c_level_ = input("HbA1c Level: ")
    blood_glucose_level_ = input("Blood Glucose Level: ")

    data = {
        'age': [age_],
        'hypertension': [hypertension_],
        'heart_disease': [heart_disease_],
        'bmi': [bmi_],
        'HbA1c_level': [HbA1c_level_],
        'blood_glucose_level': [blood_glucose_level_]
    }
    data_df = pd.DataFrame(data)

    # load scaler and normalize input data
    scaler = joblib.load('res/scaler.pkl')
    data_df_norm = scaler.transform(data_df)

    # load model and scaler from file
    model = joblib.load('res/model_diagnostics.pkl')

    # predict if diabetes yes (1) or no (0)
    predictions = model.predict(data_df_norm)  # X_new is your new data
    if predictions[0] == 0:
        result = '---> RESULT: DIABETES NEGATIVE'
    else:
        result = '---> RESULT: DIABETES POSITIVE'
    print(result)