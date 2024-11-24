from zenml import pipeline
from steps.data_ingestion_step import data_ingestion_step
from steps.handle_missing_values_step import handle_missing_values_step
from steps.feature_engineering_step import feature_engineer_step
from steps.oversampling_step import oversample_step
from steps.outlier_detection_step import outlier_detection_step
from steps.data_splitter_step import data_splitter_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step
from zenml import Model

@pipeline(
        model=Model(
            name="loan_approval_predictor"
        ),
)
def loan_approval_pipeline():
    raw_data = data_ingestion_step(
        file_path = "/home/kenzi/loan approval system zenml/data/Dataset.zip"
        )

    filled_data = handle_missing_values_step(raw_data)

    engineered_data = feature_engineer_step(
        filled_data, strategy="log", features=['LoanAmount']
    )

    engineered_data2 = feature_engineer_step(
        engineered_data, strategy="onehot_encoding", features=['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status']
    )

    engineered_data3 = feature_engineer_step(
        engineered_data2, strategy="label_encoding", features=['Property_Area', 'Dependents']
    )

    # engineered_data4 = feature_engineer_step(
    #     engineered_data3, strategy="minmax_scaling", features=['LoanAmount']
    # )

    # capped_data = outlier_detection_step(
    #     engineered_data3, column_name="LoanAmount"
    # )
    oversampled_data = oversample_step(engineered_data3, feature='Loan_Status_Y')

    X_train, X_test, y_train, y_test = data_splitter_step(
        oversampled_data,
        target_column="Loan_Status_Y"
    )
    model = model_building_step(X_train=X_train, y_train=y_train)
    evaluation_metrics, mse = model_evaluator_step(
        trained_model=model, X_test=X_test, y_test=y_test
    )

    return model

    
if __name__ == "__main__":
    run = loan_approval_pipeline()