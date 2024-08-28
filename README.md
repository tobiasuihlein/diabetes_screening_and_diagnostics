# Diabetes Prediction

## Introduction
This project aims to predict whether a person has diabetes based on various health-related attributes. The dataset used for this project is sourced from Kaggle and contains several features that are used to build and evaluate machine learning models.

## Dataset
The dataset can be found [here](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset/data) on Kaggle. It contains the following features:

* Age: Age of the patient (numerical)
* Gender: Gender of the patient (categorical)
* BMI: Body mass index (numerical)
* Hypertension: Presence of hypertension (boolean)
* HeartDisease: Presence of heart disease (boolean)
* SmokingHistory: History of smoking (categorical)
* HbA1cLevel: Hemoglobin A1c level (numerical)
* BloodGlucoseLevel: Blood glucose level (numerical)
* Diabetes: Diabetes status (boolean)

## Data cleaning and Feature engineering
The dataset was largely cleaned; however, we removed null values and addressed the median spike in BMI. After evaluating the correlation coefficients, we found that gender and smoking history had no significant impact on the diabetes outcome. Consequently, we decided to remove those columns from the dataset.

## Sensitivity and Specificity
* Sensitivity (true positive rate) is the probability of a positive test result, conditioned on the individual truly being positive.<br>
* Specificity (true negative rate) is the probability of a negative test result, conditioned on the individual truly being negative.

## Model Building and Evaluation (without oversampling)
Given that only 11% of the dataset included individuals with diabetes, we implemented an oversampling technique. This approach led to an increase in specificity but a decrease in sensitivity. We evaluated various algorithms both with and without the application of oversampling.


<img width="707" alt="Screenshot 2024-08-08 at 16 46 26" src="https://github.com/user-attachments/assets/c4c08f38-2bbe-45c4-aa67-89173cac99fd">


##  Hyperparameter tuning (without oversampling)

<img width="772" alt="Screenshot 2024-08-08 at 16 51 10" src="https://github.com/user-attachments/assets/8381f554-880c-4a05-b889-1571f971d628">

##  Hyperparameter tuning (with oversampling)

<img width="777" alt="Screenshot 2024-08-09 at 08 31 23" src="https://github.com/user-attachments/assets/e9c29738-1e2e-4f57-8a31-db4bac1cfcc8">





## Conclusion
* Filling missing values with median can be a valid option
* Higher number of trees does not automatically improve the model
* Oversampling can improve sensitivity while decreasing specificity


## Additional links
https://docs.google.com/presentation/d/1bGwimebjv2tiCOu6i-ZmnrG9b8XboAUu5IwIYmmtOVw/edit#slide=id.g2f140340185_0_168



