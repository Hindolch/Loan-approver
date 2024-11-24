import logging
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract base class for model evaluation strategy
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self, model:ClassifierMixin, X_test: pd.DataFrame, y_test:pd.Series)->dict:
        """
        Abstract method to evaluate a model

        Parameters:
        model(Regression): The trained model to evaluate
        X_test(pd.DataFrame): The testing data features.
        y_test(pd.Series): The testing data labels/targets

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        pass

# Concrete strategy for SVM model evaluation
class RFmodelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model, X_test:pd.DataFrame, y_test:pd.Series)->dict:
        """
        Evaluates a regression model using accuracy score and precision score.

        Parameters:
        model (RegressorMixin): The trained regression model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing accuracy score and precision score.
        """
        logging.info("Predicting using the trained model")
        y_pred = model.predict(X_test)

        logging.info("Calculating evaluation metrics")
        accu_score = accuracy_score(y_test, y_pred)
        preci_score = precision_score(y_test, y_pred)

        metrics = {"Accuracy score": accu_score, "Precision score:": preci_score}
        logging.info(f"Model Evaluation metrics: {metrics}")
        return metrics

# Context class for model evaluation
class ModelEvaluator:
    def __init__(self, strategy:ModelEvaluationStrategy):
        """
        Initializes the ModelEvaluator with a specific model evaluation strategy

        Parameters:
        strategy(ModelEvaluationStrategy): The strategy to be used for model evaluation
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy:ModelEvaluationStrategy):
        """
        Sets a new strategy for the ModelEvaluator

        Parameters:
        strategy(ModelEvaluationStrategy): The strategy to be used for model evaluation.
        """
        logging.info("Switching model evaluation strategy")
        self._strategy = strategy
    
    def evaluate(self, model:ClassifierMixin, X_test:pd.DataFrame, y_test:pd.Series)->dict:
        """
        Executes the model evaluation using the current strategy.

        Parameters:
        model (ClassifierMixin): The trained model to evaluate.
        X_test (pd.DataFrame): The testing data features.
        y_test (pd.Series): The testing data labels/target.

        Returns:
        dict: A dictionary containing evaluation metrics.
        """
        logging.info("Evaluating the model using the selected strategy")
        return self._strategy.evaluate_model(model,X_test,y_test)


# Example usage
if __name__ == "__main__":
    # Example trained model and data (replace with actual trained model and data)
    # model = trained_sklearn_model
    # X_test = test_data_features
    # y_test = test_data_target

    # Initialize model evaluator with a specific strategy
    # model_evaluator = ModelEvaluator(RegressionModelEvaluationStrategy())
    # evaluation_metrics = model_evaluator.evaluate(model, X_test, y_test)
    # print(evaluation_metrics)

    pass