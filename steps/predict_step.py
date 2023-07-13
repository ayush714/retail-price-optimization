import numpy as np  # type: ignore [import]
from rich import print as rich_print  # type: ignore [import]
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.steps import Output


@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
) -> Output(predictions=np.ndarray):
    """Run a inference request against a prediction service."""
    service.start(timeout=60)  # should be a NOP if already started
    prediction = service.predict(data)
    rich_print("Prediction: ", prediction)
    return prediction