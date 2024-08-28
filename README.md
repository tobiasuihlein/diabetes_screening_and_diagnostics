# Diabetes Screening vs. Diagnostics
#### Contributors
* Kaveri Chetia
* Tobias Uihlein

#### Techniques and Tools used in this project
* Python including various libraries (Pandas, SkLearn, Seaborn)
* Machine Learning Model Training and Testing
* Oversampling
* Hyperparameter tuning


## Introduction
This project aims to predict whether a person has diabetes based on various health-related attributes.
The dataset used for this project is sourced from Kaggle and contains several features that are used to build and evaluate different machine learning models.
In particular, the models and the data are tested whether they are better suited for diagnostics or screening purposes.

## Dataset
The dataset can be found [here](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data). It contains the following features:

* Age: Age of the patient (numerical)
* Gender: Gender of the patient (categorical)
* BMI: Body mass index (numerical)
* Hypertension: Presence of hypertension (boolean)
* HeartDisease: Presence of heart disease (boolean)
* SmokingHistory: History of smoking (categorical)
* HbA1cLevel: Hemoglobin A1c level (numerical)
* BloodGlucoseLevel: Blood glucose level (numerical)
* Diabetes: Diabetes status (boolean)

## Data Cleaning and Feature Engineering
The dataset was largely cleaned by source; however, we removed null values and addressed a spike in the BMI distribution. After evaluating the correlation coefficients, we found that gender and smoking history had no significant impact on the diabetes outcome. Consequently, we decided to remove those columns from the dataset. Here, we tune and test the models for both characteristics.

## Sensitivity and Specificity
In medicine one differentiates between screening and diagnostic tools. While the former requires a high sensitivity (catch all possible true cases), the latter requires a high specificity (catch only true cases).
* Sensitivity (true positive rate) is the probability of a positive test result, conditioned on the individual truly being positive.<br>
* Specificity (true negative rate) is the probability of a negative test result, conditioned on the individual truly being negative.

## Model Building and Evaluation
We evaluated various algorithms and found Random Forest and Gradient Boosting to perform best in overall accuracy aswell as in sensitivity and specificity.

<img width="707" alt="Screenshot 2024-08-08 at 16 46 26" src="https://github.com/user-attachments/assets/c4c08f38-2bbe-45c4-aa67-89173cac99fd">

##  Hyperparameter tuning without oversampling
Next we tested the two models with different hyperparameters.
We found parameter values providing a model with 99.9% specificity, which makes the model well suited for diagnostics.
The highest sensitivity we could find, however, was only around 73%.

<img width="772" alt="Screenshot 2024-08-08 at 16 51 10" src="https://github.com/user-attachments/assets/8381f554-880c-4a05-b889-1571f971d628">

##  Hyperparameter tuning with oversampling
Given that only 11% of the dataset included individuals with diabetes, we implemented an oversampling technique.
Again testing different hyperparameter values, we found a model performing with a sensitivity of 91.4%.

<img width="777" alt="Screenshot 2024-08-09 at 08 31 23" src="https://github.com/user-attachments/assets/e9c29738-1e2e-4f57-8a31-db4bac1cfcc8">

## Conclusion
* We found a model well suited for diagnostic purposes with 99.9% specificity
* Using oversampling we could improve the sensitivity to up to 91.4% while decreasing specificity


## Additional links
* [Presentation Slides](https://docs.google.com/presentation/d/1bGwimebjv2tiCOu6i-ZmnrG9b8XboAUu5IwIYmmtOVw/edit#slide=id.g2f140340185_0_168)
* [Dataset on Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data)



