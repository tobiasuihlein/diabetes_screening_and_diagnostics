import pandas as pd
from sklearn.metrics import accuracy_score, recall_score


def model_testing(X_train, X_test, y_train, y_test, model_dict):

    model_types_list = []
    accuracies = []
    sensitivities = []
    specificities = []

    #loop over models
    for model_type_name, model_type in model_dict.items():
        print(f'Model type: {model_type_name}')
        # Create model
        model = model_type

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Calculate sensitivity [tp / (tp + fn)] (= recall)
        sensitivity = recall_score(y_test, y_pred)

        # Calculate specificity  [tn / (tn + fp)]
        specificity = recall_score(y_test, y_pred, pos_label=0)

        model_types_list.append(model_type_name)
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    df_result = pd.DataFrame({'model_type': model_types_list,
                              'accuracy': accuracies,
                              'sensitivity': sensitivities,
                              'specificity': specificities})
    
    return df_result


def hyperparameter_tuning(X_train, X_test, y_train, y_test, model_dict, n_estimators, max_depth):
    n_estimators_grid = n_estimators
    max_depth_grid = max_depth

    model_types_list = []
    n_estimators_list = []
    max_depth_list = []
    accuracies = []
    sensitivities = []
    specificities = []

    #loop over models
    for model_type_name, model_type in model_dict.items():
        print(f'Model type: {model_type_name}')
        # loop over parameter grid
        for n_estimators in n_estimators_grid:
            print(f'n_estimators: {n_estimators}')
            for max_depth in max_depth_grid:
                print(f'max_depth: {max_depth}')
                # Create model
                model = model_type(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)

                # Calculate sensitivity [tp / (tp + fn)] (= recall)
                sensitivity = recall_score(y_test, y_pred)

                # Calculate specificity  [tn / (tn + fp)]
                specificity = recall_score(y_test, y_pred, pos_label=0)

                model_types_list.append(model_type_name)
                n_estimators_list.append(n_estimators)
                max_depth_list.append(max_depth)
                accuracies.append(accuracy)
                sensitivities.append(sensitivity)
                specificities.append(specificity)

    df_result = pd.DataFrame({'model_type': model_types_list,
                                'n_estimators': n_estimators_list,
                                'max_depth': max_depth_list,
                                'accuracy': accuracies,
                                'sensitivity': sensitivities,
                                'specificity': specificities})
    
    return df_result