# all common functionalities

import sys
import os

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path,obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok =True)

    with open(file_path,"wb") as file_obj:
      dill.dump(obj,file_obj)
  except Exception as e:
    raise CustomException(e,sys)
  
def evaluate_models(X_train,y_train,x_test,y_test,models):
    try:
        report = {}
        
        params = {
            "Random Forest": {
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Decision Tree": {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
            },
            "Gradient Boosting": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Linear Regression": {},
            "K-Neighbors Regressor": {
                'n_neighbors': [5, 7, 9, 11]
            },
            "XGBRegressor": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "CatBoosting Regressor": {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            },
            "AdaBoost Regressor": {
                'learning_rate': [0.1, 0.01, 0.5, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }
        
        for i in range(len(list(models))):
            
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            # Train model
            # model.fit(X_train,y_train)

            # Predict Testing data
            y_train_pred = model.predict(X_train)
            
            y_test_pred = model.predict(x_test)

            # Get R2 scores for train and test data
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)