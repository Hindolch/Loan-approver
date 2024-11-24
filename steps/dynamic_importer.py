# import pandas as pd
# import numpy as np
# from zenml import step
# from sklearn.preprocessing import LabelEncoder

# @step
# def dynamic_importer() -> str:
#     """
#     Dynamically imports synthetic data matching the specific column requirements 
#     for the loan approval model with label-encoded Property_Area.
#     """
 
    
#     data = {
#         # Numerical columns with realistic ranges
#         'ApplicantIncome': [5678.0, 21213.0],
#         'CoapplicantIncome': [4599.0, 2133.0],
#         'LoanAmount': [567.0, 34.0],
#         'Loan_Amount_Term': [180, 360],
#         'Credit_History': [1, 0],
        
#         # Binary encoded categorical columns
#         'Gender_Male': [0, 1],
#         'Married_Yes': [1, 1],
#         'Education_Not Graduate': [1, 1],
#         'Self_Employed_Yes': [1, 0],
        
#         # Label encoded Property Area
#         'Property_Area': [1,2],
#         'Dependents': [2, 0]
#     }
    
#     # Create DataFrame
#     df = pd.DataFrame(data)
    
#     # # Ensure data types are correct
#     # df = df.astype({
#     #     'ApplicantIncome': float,
#     #     'CoapplicantIncome': float,
#     #     'LoanAmount': float,
#     #     'Loan_Amount_Term': int,
#     #     'Credit_History': int,
#     #     'Gender_Male': int,
#     #     'Married_Yes': int,
#     #     'Education_Not Graduate': int,
#     #     'Self_Employed_Yes': int,
#     #     'Property_Area': int,
#     #     'Dependents': int
#     # })

#     # Convert the DataFrame to a JSON string
#     json_data = df.to_json(orient="split")
    
#     print(json_data)
#     return json_data

# if __name__ == "__main__":
#     dynamic_importer()


import pandas as pd
from zenml import step

@step
def dynamic_importer() -> str:
    """
    Dynamically imports synthetic data matching the specific column requirements 
    for the loan approval model.
    """
    # Define synthetic data
    data = {
        'ApplicantIncome': [5678.0, 21213.0],
        'CoapplicantIncome': [4599.0, 2133.0],
        'LoanAmount': [567.0, 34.0],
        'Loan_Amount_Term': [180, 360],
        'Credit_History': [1, 0],
        'Gender_Male': [0, 1],
        'Married_Yes': [1, 1],
        'Education_Not Graduate': [1, 1],
        'Self_Employed_Yes': [1, 0],
        'Property_Area': [1, 2],
        'Dependents': [2, 0]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Convert the DataFrame to a JSON string with "split" orientation
    json_data = df.to_json(orient="split")

    # Debugging: Print JSON data structure
    print("Generated JSON Data:", json_data)
    
    return json_data

if __name__ == "__main__":
    dynamic_importer()
