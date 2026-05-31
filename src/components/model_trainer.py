import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor  #type: ignore
from sklearn.ensemble import ( # type: ignore
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import (  #type: ignore
    LinearRegression,
    Lasso,
    Ridge
)
from sklearn.neighbors import KNeighborsRegressor  #type: ignore
from sklearn.tree import DecisionTreeRegressor  #type: ignore
from sklearn.metrics import r2_score  #type: ignore
from xgboost import XGBRegressor #type: ignore

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    training_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            ### Best model score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found on both training and testing dataset: {best_model_name} with r2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.training_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)