from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW, TENSORFLOW

from steps.ingest_data import ingest

# from steps.predict_processor_step import predict_preprocessor
from steps.predict_step import predictor
from steps.prediction_service_loader_step import prediction_service_loader

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(enable_cache=True, settings={"docker": docker_settings})
def infer_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = ingest("retail_prices_processed", for_predict=True)
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(model_deployment_service, batch_data)