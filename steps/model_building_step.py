from zenml import step
import pandas as pd
from sklearn.base import ClassifierMixin
import logging
from typing import Annotated
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from zenml import ArtifactConfig
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import mlflow
from zenml.client import Client

# Get the active experiment tracker from ZenML
experiment_tracker = Client().active_stack.experiment_tracker
from zenml import Model

model = Model(
    name="loan_approval_predictor",
    version=None,
    license="Apache 2.0",
    description="Loan approval prediction model for individuals.",
)


@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    """
    Builds and trains a Random Forest Classifier model using scikit-learn wrapped in a pipeline.

    Parameters:
    X_train (pd.DataFrame): The training data features.
    y_train (pd.Series): The training data labels/targets.

    Returns:
    Pipeline: The trained scikit-learn pipeline including preprocessing and the Random Forest model.
    """
    # Print out all columns for debugging
    print("Columns in X_train:")
    print(X_train.columns.tolist())

    # Dynamically determine column groups based on actual DataFrame columns
    one_hot_cols = [col for col in ['Gender_Male', 'Married_Yes', 'Education_Not Graduate', 'Self_Employed_Yes'] 
                    if col in X_train.columns]
    
    label_encode_cols = [col for col in ['Property_Area', 'Dependents'] 
                         if col in X_train.columns]
    
    numerical_cols = [col for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                                      'Loan_Amount_Term', 'Credit_History'] 
                      if col in X_train.columns]

    logging.info(f"One-hot encoding columns: {one_hot_cols}")
    logging.info(f"Label encoding columns: {label_encode_cols}")
    logging.info(f"Numerical columns: {numerical_cols}")

    # Define label encoding transformation
    def label_encode_transformer(df: pd.DataFrame):
        df_encoded = df.copy()
        for col in label_encode_cols:
            df_encoded[col] = LabelEncoder().fit_transform(df[col])
        return df_encoded

    label_encoder = FunctionTransformer(label_encode_transformer)

    # Define preprocessing transformers
    numerical_transformer = SimpleImputer(strategy="median")
    categorical_onehot_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine transformations in ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat_onehot", categorical_onehot_transformer, one_hot_cols),
            ("cat_label", label_encoder, label_encode_cols),
        ],
        remainder="passthrough",
    )

    # Define the model
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    #model = XGBClassifier(n_estimators=300,learning_rate=0.1,max_depth=5,random_state=42,use_label_encoder=False)

    # Create a complete pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    # Start an MLflow run to log the model training process
    if not mlflow.active_run():
        mlflow.start_run()  # Start a new MLflow run if there isn't one active


    # Train the pipeline
    try:
        # Enable autologging for scikit-learn to automatically capture model metrics, parameters, and artifacts
        mlflow.sklearn.autolog()

        logging.info("Building and training the Random Forest model.")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")
         # Log the columns that the model expects
        expected_columns = numerical_cols + one_hot_cols + label_encode_cols
        logging.info(f"Model expects the following columns: {expected_columns}")

    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e

    finally:
        mlflow.end_run()
        
    return pipeline