from zenml import step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )

    return existing_services[0]