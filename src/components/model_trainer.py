import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], #All columns except the last one
                train_array[:, -1], #Last column as target variable
                test_array[:, :-1], #All columns except the last one
                test_array[:, -1], #Last column as target variable
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
                "K-Neighnors Classifier": KNeighborsRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=0),
                "XGBoost Classifier": XGBRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64],
                    #'criterion': ['squared_error', 'absolute_error', 'poisson'],
                },
                "Gradient Boosting": {
                    #'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.6, 0.7, 0.75, 0.8,0.85,0.9],
                    'n_estimators': [50, 100, 200],
                },
                "AdaBoost Classifier": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                },
                "K-Neighnors Classifier": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                },
                "Linear Regression": {},
                "CatBoosting Classifier": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100],
                },
                "XGBoost Classifier": {
                    
                    'learning_rate': [0.01, 0.1, 0.05,0.001],
                    'n_estimators': [8, 16, 32,64,128,256]
                }
            }

            model_report:dict = evaluate_model(
                models=models,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                params=params
            )

           

            best_model_score = max(sorted(model_report.values()))
            
            best_model_name =list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys) from e