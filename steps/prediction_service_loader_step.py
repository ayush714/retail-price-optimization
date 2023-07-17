# from zenml import step
# from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
#     MLFlowModelDeployer,
# )
# from zenml.integrations.mlflow.services import MLFlowDeploymentService


# @step(enable_cache=False)
# def prediction_service_loader(
#     pipeline_name: str,
#     pipeline_step_name: str,
#     running: bool = True,
#     model_name: str = "model",
# ) -> MLFlowDeploymentService:
#     """Get the prediction service started by the deployment pipeline.

#     Args:
#         pipeline_name: name of the pipeline that deployed the MLflow prediction
#             server
#         step_name: the name of the step that deployed the MLflow prediction
#             server
#         running: when this flag is set, the step only returns a running service
#         model_name: the name of the model that is deployed
#     """
#     # get the MLflow model deployer stack component
#     model_deployer = MLFlowModelDeployer.get_active_model_deployer()

#     # fetch existing services with same pipeline name, step name and model name
#     existing_services = model_deployer.find_model_server(
#         pipeline_name=pipeline_name,
#         pipeline_step_name=pipeline_step_name,
#         model_name=model_name,
#         running=running,
#     )

#     if not existing_services:
#         raise RuntimeError(
#             f"No MLflow prediction service deployed by the "
#             f"{pipeline_step_name} step in the {pipeline_name} "
#             f"pipeline for the '{model_name}' model is currently "
#             f"running."
#         )

#     return existing_services[0]


from typing import cast

from zenml import step
from zenml.integrations.bentoml.model_deployers.bentoml_model_deployer import (
    BentoMLModelDeployer,
)
from zenml.integrations.bentoml.services.bentoml_deployment import (
    BentoMLDeploymentService,
)


@step(enable_cache=False)
def bentoml_prediction_service_loader(
    pipeline_name: str, step_name: str, model_name: str
) -> BentoMLDeploymentService:
    """Get the BentoML prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the model.
        step_name: the name of the step that deployed the model.
        model_name: the name of the model that was deployed.
    """
    model_deployer = BentoMLModelDeployer.get_active_model_deployer()

    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        model_name=model_name,
    )
    if not services:
        raise RuntimeError(
            f"No BentoML prediction server deployed by the "
            f"'{step_name}' step in the '{pipeline_name}' "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )

    if not services[0].is_running:
        raise RuntimeError(
            f"The BentoML prediction server last deployed by the "
            f"'{step_name}' step in the '{pipeline_name}' "
            f"pipeline for the '{model_name}' model is not currently "
            f"running."
        )

    return cast(BentoMLDeploymentService, services[0])