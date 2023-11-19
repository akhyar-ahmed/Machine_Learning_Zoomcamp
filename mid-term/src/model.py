
import numpy as np
import logging

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# Perform hyperparameter tunning and return the best model
def do_hyperparameter_tuning(features, labels, model_name):
    model = None
    parameters = None
    if model_name == "RFClassifier":
        # Create the Random Forest Classifier model
        model = RandomForestClassifier(max_features = "sqrt")
        # Define the parameter grid to search
        parameters = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
    elif model_name == "MLPClassifier":
        # Create the MLP Classifier model
        model = MLPClassifier()
        # Define the parameter grid to search
        parameters = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50, 25)],
            'activation': ["relu", "tanh", "logistic"],
            'alpha': [0.0001, 0.001, 0.01],
        }
    elif model_name == "LogisticRegression":
        # Create the Logistic Regression Classifier model
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        # Define the parameter grid to search
        parameters = {
            "penalty": ['l1', 'l2'],
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
        }
    elif model_name == "XGBClassifier":
        # Create the XGB Classifier model
        model = XGBClassifier()
        # Define the parameter grid to search
        parameters = {
            "learning_rate": [0.001, 0.01, 0.1, 0.2, 1.0],
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [1, 3, 4, 5],
        }       


    cv = GridSearchCV(model, parameters, cv = 5, scoring = "f1", n_jobs=-1)
    cv.fit(features, labels)

    best_model = cv.best_estimator_
    best_score = round(cv.best_score_, 2)
    logging.info(f"Best estimator for {model_name}_model:\n {best_model}")
    logging.info(f"Best scores for {model_name}_model:\n {best_score}")
    return best_model


# Get the model
def get_model(model_name):
    if model_name == "RFClassifier":
        # Create the Random Forest Classifier model
        model = RandomForestClassifier()
    elif model_name == "MLPClassifier":
        # Create the MLP Classifier model
        model = MLPClassifier()
    elif model_name == "LogisticRegression":
        # Create the Logistic Regression Classifier model
        model = LogisticRegression()
    elif model_name == "XGBClassifier":
        # Create the XGB Classifier model
        model = XGBClassifier()
    return model