from typing import List, Tuple

import mlflow
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from zenml.integrations.mlflow.experiment_trackers import MLFlowExperimentTracker
from zenml.logger import get_logger

from materializer.custom_materializer import ListMaterializer, StatsModelMaterializer
from steps.src.model_building import LinearRegressionModel, ModelRefinement

logger = get_logger(__name__)
experiment_tracker = Client().active_stack.experiment_tracker

if not experiment_tracker or not isinstance(
    experiment_tracker, MLFlowExperimentTracker
):
    raise RuntimeError(
        "Your active stack needs to contain a MLFlow experiment tracker for "
        "this example to work."
    )

@step(experiment_tracker="mlflow_experiment_tracker",
  settings={"experiment_tracker.mlflow": {"experiment_name": "test_name"}}, output_materializers=[StatsModelMaterializer, ListMaterializer])
def train(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.Series, "y_train"]
) -> Tuple[
    Annotated[RegressionResultsWrapper, "model"],
    Annotated[List[str], "predictors"],
]:
    """Trains a linear regression model and outputs the summary."""
    try:
        print(y_train)
        model = LinearRegressionModel(X_train, y_train)
        mlflow.statsmodels.autolog()
        model = model.train()  
        df = pd.concat([X_train, y_train], axis=1)
        refinement1 = ModelRefinement(model, df)
        predictors = refinement1.remove_insignificant_vars(alpha=0.05)  # removes insignificant variables 
        return model, predictors
    except Exception as e:
        logger.error(e)
        raise e


@step(experiment_tracker="mlflow_experiment_tracker",
  settings={"experiment_tracker.mlflow": {"experiment_name": "test_name"}})
def re_train(
    X_train: Annotated[pd.DataFrame, "X_train"],
    y_train: Annotated[pd.Series, "y_train"], 
    predictors: list
) -> Tuple[
    Annotated[RegressionResultsWrapper, "model"],
    Annotated[pd.DataFrame, "df_with_significant_vars"],
]:
    """Trains a linear regression model and outputs the summary."""
    try:  
        print(X_train[predictors])
        model = LinearRegressionModel(X_train[predictors], y_train)
        mlflow.statsmodels.autolog()
        model = model.train() 
        df_with_significant_vars = pd.concat([X_train[predictors], y_train], axis=1)  
        df_with_significant_vars.rename(columns={"series": 'qty'}, inplace=True) 
        # df_with_significant_vars.to_csv("df_with_significant_vars.csv", index=False)
        logger.info("Model trained successfully")
        return model, df_with_significant_vars
    except Exception as e:
        logger.error(e)
        raise e 