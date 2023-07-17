# from zenml import pipeline
# from zenml.config import DockerSettings
# from zenml.integrations.constants import MLFLOW, TENSORFLOW

# from steps.ingest_data import ingest

# # from steps.predict_processor_step import predict_preprocessor
# from steps.predict_step import predictor
# from steps.prediction_service_loader_step import prediction_service_loader

# docker_settings = DockerSettings(required_integrations=[MLFLOW])


# @pipeline(enable_cache=True, settings={"docker": docker_settings})
# def infer_pipeline(pipeline_name: str, pipeline_step_name: str):
#     # Link all the steps artifacts together
#     batch_data = ingest("retail_prices_processed", for_predict=True)
#     model_deployment_service = prediction_service_loader(
#         pipeline_name=pipeline_name,
#         pipeline_step_name=pipeline_step_name,
#         running=False,
#     )
#     predictor(model_deployment_service, batch_data)


from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import BENTOML, PYTORCH

from steps.ingest_data import ingest
from steps.predict_step import predictor
from steps.prediction_service_loader_step import bentoml_prediction_service_loader

docker_settings = DockerSettings(required_integrations=[PYTORCH, BENTOML])


@pipeline(settings={"docker": docker_settings})
def inference_fashion_mnist(
    model_name: str, pipeline_name: str, step_name: str
):
    """Perform inference with a model deployed through BentoML.

    Args:
        pipeline_name: The name of the pipeline that deployed the model.
        step_name: The name of the step that deployed the model.
        model_name: The name of the model that was deployed.
    """
    inference_data = ingest(table_name="retail_prices", for_predict=True)
    prediction_service = bentoml_prediction_service_loader(
        model_name=model_name, pipeline_name=pipeline_name, step_name=step_name
    )
    predictor(inference_data=inference_data, service=prediction_service)