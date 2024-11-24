from zenml import step
from src.model_evaluator import ModelEvaluator, RFmodelEvaluationStrategy
import logging
from typing import Tuple
import pandas as pd
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


@step(enable_cache=False)
def model_evaluator_step(
    trained_model:Pipeline, X_test:pd.DataFrame, y_test:pd.Series
)->Tuple[dict,float]:
    """
    Evaluates the trained model using ModelEvaluator and SVMmodelEvaluationStrategy.

    Parameters:
    trained_model (Pipeline): The trained pipeline containing the model and preprocessing steps.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels/target.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    # Print out all columns for debugging
    print("Columns in X_test:")
    print(X_test.columns.tolist())

    # Ensure the inputs are of the correct types
    if not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_test must be a pandas dataframe")
    if not isinstance(y_test, pd.Series):
        raise ValueError("y_test must be a pandas series")
    
    logging.info("Preparing to evaluate the model")

    # Dynamically determine column groups
    categorical_cols = [col for col in X_test.columns 
                        if X_test[col].dtype in ['object', 'category']]
    
    numerical_cols = [col for col in X_test.columns 
                      if X_test[col].dtype not in ['object', 'category']]

    logging.info(f"Categorical columns: {categorical_cols}")
    logging.info(f"Numerical columns: {numerical_cols}")

    try:
        # Use the preprocessor from the trained model to transform test data
        X_test_preprocessed = trained_model.named_steps["preprocessor"].transform(X_test)
        logging.info("Successfully preprocessed test data")
    except Exception as e:
        logging.error(f"Error during test data preprocessing: {e}")
        raise ValueError(f"Failed to preprocess test data: {e}")

    # Initialize the evaluator and model prediction
    evaluator = ModelEvaluator(strategy=RFmodelEvaluationStrategy())

    # Perform the evaluation
    try:
        evaluation_metrics = evaluator.evaluate(
            trained_model.named_steps["model"], X_test_preprocessed, y_test
        )
        logging.info("Model evaluation completed")
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise ValueError(f"Failed to evaluate model: {e}")

    # Validate and extract accuracy
    if not isinstance(evaluation_metrics, dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    
    accuracy_score = evaluation_metrics.get("Accuracy score", None)
    
    if accuracy_score is None:
        logging.warning("No accuracy score found in evaluation metrics")
    
    return evaluation_metrics, accuracy_score