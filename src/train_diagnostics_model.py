print("- import packages")
import sys
import os
import pandas as pd
import joblib

if __name__ == "__main__":

    ##### LOAD AND CLEAN RAW DATA #####
    print("- loading and cleaning dataset")

    # load original data frame
    df = pd.read_csv("../data/diabetes_prediction_dataset.csv")

    # drop missing values
    df=df.loc[df['smoking_history'] != 'No Info']
    df=df.loc[df['gender'] != 'Other']

    # aggregate values for smoking history
    df.loc[df['smoking_history'] == 'not current', 'smoking_history'] = 'former'
    df.loc[df['smoking_history'] == 'ever', 'smoking_history'] = 'never'


    #### FEATURE ENGINEERING ####
    print("- perform feature engineering")

    # transform categorical columns to numerical
    df["smoking_history"] = df["smoking_history"].map({'never': 0, 'former': 1, 'current': 2})
    df["gender"] = df["gender"].map({'Male': 0, 'Female': 1})
    df = df.rename(columns={'gender': 'gender_female'})

    # drop unnecessary columns
    df = df.drop(columns = ['smoking_history', 'gender_female'])


    #### DATA PREPARATION ####
    print("- prepare data for model training")

    # define features and target
    features = df.drop(columns = ['diabetes'])
    target = df['diabetes']

    # normalize features
    from sklearn.preprocessing import StandardScaler
    normalizer = StandardScaler()
    features_norm = normalizer.fit_transform(features)


    #### TRAIN AND SAVE MODEL ####
    print("- train and apply model")

    # import model
    from sklearn.ensemble import RandomForestClassifier

    # set hyperparameters
    n_estimators = 100
    max_depth = 5

    # create and train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(features_norm, target)

    # save trained model and scaler
    print("- save model and scaler")
    joblib.dump(model, 'res/model_diagnostics.pkl')
    joblib.dump(normalizer, 'res/scaler.pkl')