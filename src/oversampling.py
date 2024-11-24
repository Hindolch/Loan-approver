import logging
from abc import ABC, abstractmethod
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base class for Random oversampling strategy
# -----------------------------------------------------
# This class defines a common interface for oversampling strategy
# Subclass must implement the apply_oversampling method
class OversamplingStrategy(ABC):
    @abstractmethod
    def apply_oversampling(self, df: pd.DataFrame)->pd.DataFrame:
        """
        Abstract method to apply oversampling to the dataframe

        Parameters:
        df(pd.DataFrame): The dataframe containing features to oversample

        Returns:
        pd.DataFrame: The oversampled dataframe
        """
        pass

# Concrete strategy for oversampling
# ------------------------------------------
# This strategy applies SMOTE oversampling to the given feature
class SmoteTransformation(OversamplingStrategy):
    def __init__(self, feature:str):
        """
        Initializes the SmoteTransformation with the specific feature to oversample

        Parameters:
        feature: The feature on which to apply oversampling
        """
        self.feature = feature

    def apply_oversampling(self, df: pd.DataFrame)->pd.DataFrame:
        """
        Applies oversampling to the specified feature in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.

        Returns:
        pd.DataFrame: The overampled dataframe.
        """
        logging.info(f"Applying SMOTE transformation to the feature: {self.feature}")
        x = df.drop(columns=[self.feature])
        y = df[self.feature]
        # Initialize SMOTE
        smote = SMOTE(random_state=42)

        # Fit and apply SMOTE
        X_balanced, y_balanced = smote.fit_resample(x, y)
        # Create new balanced dataframe
        df_oversampled = pd.concat([pd.DataFrame(X_balanced, columns=x.columns), 
                        pd.Series(y_balanced, name=self.feature)], axis=1)
        return df_oversampled
    

# Context class for oversamping
# This class uses an oversampling strategy to apply oversampling
class OverSampler:
    def __init__(self, strategy:OversamplingStrategy):
        """
        Initialized the oversampler with a specific oversampling strategy

        Parameters:
        strategy(OversamplingStrategy): The strategy to be used for oversampling the feature
        """
        self._strategy = strategy

    def set_strategy(self, strategy:OversamplingStrategy):
        """
        Sets a new strategy for OverSampler

        Parameters:
        strategy(OverSamplingStrategy): The new strategy to be used for oversampling
        """
        logging.info("Switching oversampling strategy")
        self._strategy = strategy

    def execute_oversampling(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Executes the oversampler using the current strategy.

        Parameters:
        df(pd.DataFrame): The dataframe contaiing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied oversampling

        """
        return self._strategy.apply_oversampling(df)    