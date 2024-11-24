import json
import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from dynamic_importer import dynamic_importer
from zenml.client import Client

@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service for loan approval.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (str): The input data as a JSON string.

    Returns:
        np.ndarray: The model's prediction.
    """

    # Start the service (should be a NOP if already started)
    service.start(timeout=10)

    # Load the input data from JSON string
    data = json.loads(input_data)

    # Extract the actual data and expected columns
    data.pop("columns", None)  # Remove 'columns' if it's present
    data.pop("index", None)  # Remove 'index' if it's present
    # Define the columns the model expects
    expected_columns = [
        'ApplicantIncome', 
        'CoapplicantIncome', 
        'LoanAmount', 
        'Loan_Amount_Term', 
        'Credit_History', 
        'Gender_Male', 
        'Married_Yes', 
        'Education_Not Graduate', 
        'Self_Employed_Yes', 
        'Property_Area', 
        'Dependents'
    ]


    # Convert the data into a DataFrame with the correct columns
    try:
        df = pd.DataFrame(data["data"], columns=expected_columns)
    except KeyError as e:
        print("Available keys in data:", data.keys())
        print("Input data:", data)
        raise ValueError(f"Failed to create DataFrame. Missing key: {e}")


    # Convert DataFrame to JSON list for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # Add debug print statements
    print("Input Data Shape:", data_array.shape)
    print("Input Data First Row:", data_array[0] if len(data_array) > 0 else "Empty")

    # Run the prediction
    try:
        prediction = service.predict(data_array)
        
        # # Add debug print for predictions
        # print("Prediction Shape:", prediction.shape)
        # print("First few predictions:", prediction[:5] if len(prediction) > 0 else "No predictions")
        
        print(prediction)
        return prediction
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise

predictor()
# if __name__ == "__main__":
    # Replace this with the correct deployment name or retrieve the active deployment dynamically
    # client = Client()
    # deployment_name = "zenml-model"
    # service = client.get_service(deployment_name)  # Retrieve the deployed service

    # if not isinstance(service, MLFlowDeploymentService):
    #     raise ValueError("The retrieved service is not an MLFlowDeploymentService instance.")

#     # # Call the predictor step with the initialized service
#     predictor(service=service, input_data=dynamic_importer())


# @step(enable_cache=False)
# def predictor(
#     service: MLFlowDeploymentService,
#     input_data: str,
# ) -> np.ndarray:
#     """Run an inference request against a prediction service for loan approval."""
#     # Start the service
#     service.start(timeout=10)

#     # Define the columns the model expects
#     expected_columns = [
#         'ApplicantIncome', 
#         'CoapplicantIncome', 
#         'LoanAmount', 
#         'Loan_Amount_Term', 
#         'Credit_History', 
#         'Gender_Male', 
#         'Married_Yes', 
#         'Education_Not Graduate', 
#         'Self_Employed_Yes', 
#         'Property_Area', 
#         'Dependents'
#     ]

#     # Load the input data from JSON string
#     data = json.loads(input_data)

#     # Convert the data into a DataFrame
#     df = pd.DataFrame(data["data"], columns=expected_columns)

#     # Prepare data for prediction (convert to numpy array)
#     X = df.values

#     # Run the prediction
#     try:
#         # MLflow often expects a specific input format
#         prediction = service.predict({"inputs": X})
#         return prediction
#     except Exception as e:
#         print(f"Prediction Error Details: {e}")
#         raise



# @step(enable_cache=False)
# def predictor(
#     service: MLFlowDeploymentService,
#     input_data: str,
# ) -> np.ndarray:
#     """Run an inference request against a prediction service for loan approval."""
#     # Start the service
#     service.start(timeout=10)

   

#     # Load the input data from JSON string
#     data = json.loads(input_data)
#     data.pop("columns", None)
#     data.pop("index", None)
#      # Define the columns the model expects
#     expected_columns = [
#         'ApplicantIncome', 
#         'CoapplicantIncome', 
#         'LoanAmount', 
#         'Loan_Amount_Term', 
#         'Credit_History', 
#         'Gender_Male', 
#         'Married_Yes', 
#         'Education_Not Graduate', 
#         'Self_Employed_Yes', 
#         'Property_Area', 
#         'Dependents'
#     ]
#     df = pd.DataFrame(data["data"], columns=expected_columns)

#     json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
#     prediction = service.predict(json_list)



#     return prediction
    

# @step(enable_cache=False)
# def predictor(
#     service: MLFlowDeploymentService,
#     input_data: str,
# ) -> np.ndarray:
#     """Run an inference request against a prediction service for loan approval."""
#     # Start the service
#     service.start(timeout=10)

#     # Load the input data from JSON string
#     data = pd.read_json(input_data, orient='split')

#     # Prepare data for prediction
#     X = data.values.tolist()  # Convert to list explicitly

#     # Run prediction
#     try:
#         prediction = service.predict({"instances": X})
#         return prediction
#     except Exception as e:
#         print(f"Prediction Error Details: {e}")
#         raise