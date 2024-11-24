from zenml import step
import pandas as pd
from src.feature_engineering import (
    FeatureEngineer,
    LogTransformation,
    MinMaxScalingTransformation,
    StandardScalingTransformation,
    OneHotEncodingTransformation,
    LabelEncodingTransformation
)





@step(enable_cache=False)
def feature_engineer_step(
    df:pd.DataFrame, strategy:str = "log", features:list = None
    )->pd.DataFrame:
    """Performs feature engineering using FeatureEngineer and a specific strategy"""

    # Ensuring features is a list, even if not provided
    if features is None:
        features = []
    
    if strategy == "log":
        engineer = FeatureEngineer(LogTransformation(features))
    elif strategy == "standard_scaling":
        engineer = FeatureEngineer(StandardScalingTransformation(features))
    elif strategy == "minmax_scaling":
        engineer = FeatureEngineer(MinMaxScalingTransformation(features))
    elif strategy == "label_encoding":
        engineer = FeatureEngineer(LabelEncodingTransformation(features))
    elif strategy == "onehot_encoding":
        engineer = FeatureEngineer(OneHotEncodingTransformation(features))
    else:
        raise ValueError(f"Unsupported feature engineering strategy: {strategy}")

    transformed_df = engineer.apply_feature_engineering(df)
    return transformed_df



