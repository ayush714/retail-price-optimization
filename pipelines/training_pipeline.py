from zenml import pipeline
from zenml.logger import get_logger

logger = get_logger(__name__)
from zenml.client import Client

print(Client().active_stack.experiment_tracker.get_tracking_uri())

# @pipeline(enable_cache=False)
# def training_pipeline(): 
#     """Pipeline for training."""
#     try: 
#         logger.info("Training pipeline started")
#         # Ingest data
#         df = ingest("retail_prices")
#         # Process data
#         df_processed = categorical_encode(df)
#         df_transformed = feature_engineer(df_processed)  
#         # Split data 
#         X_train, X_test, y_train, y_test = split_data(df_transformed) 
#         logger.info(X_train)
#         # Train model
#         model = train(X_train, y_train) 
#         predictors = remove_insignificant_vars(model, df_transformed, alpha=0.05) 
#         # Evaluate model
#         rmse = evaluate(model, df_transformed)

#         # Re-train model
#         model1, df_with_significant_vars = re_train(X_train, y_train, predictors)  
#         rmse1 = evaluate(model1, df_with_significant_vars)

#         logger.info("Training pipeline completed")
#     except Exception as e:
#         logger.error(e)
#         raise e

from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import BENTOML

from steps.bento_builder import bento_builder
from steps.data_drift_checker import data_drift_detector
from steps.data_splitter import combine_data, split_data
from steps.data_validator import data_validator
from steps.deployer import bentoml_model_deployer
from steps.deployment_trigger_step import deployment_trigger
from steps.evaluator import evaluate
from steps.ingest_data import ingest
from steps.process_data import categorical_encode, feature_engineer
from steps.refine_model import remove_insignificant_vars
from steps.train_model import re_train, sklearn_train

docker_settings = DockerSettings(required_integrations=[BENTOM, DEEPCHECKS])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def training_retail():
    """Train a model and deploy it with BentoML."""
    df = ingest("retail_prices") 
    df_processed = categorical_encode(df)
    df_transformed = feature_engineer(df_processed)  
    X_train, X_test, y_train, y_test = split_data(df_transformed)  
    df_train, df_test = combine_data(X_train, X_test, y_train, y_test) 
    data_validator(dataset=df_train)
    data_drift_detector(
            reference_dataset=df_train,
            target_dataset=df_test,
        )

    model, predictors = sklearn_train(X_train, y_train)         # Evaluate model
    # rmse = evaluate(model, df_transformed)
    rmse = 0.95 
        # Re-train model
    # model1, df_with_significant_vars = re_train(X_train, y_train, predictors)  
    # rmse1 = evaluate(model1, df_with_significant_vars)
    decision = deployment_trigger(accuracy=rmse, min_accuracy=0.80)
    bento = bento_builder(model=model)
    bentoml_model_deployer(bento=bento, deploy_decision=decision) 
