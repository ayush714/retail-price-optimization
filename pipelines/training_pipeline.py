from zenml import pipeline
from zenml.logger import get_logger

from steps.data_splitter import split_data
from steps.evaluator import evaluate
from steps.ingest_data import ingest
from steps.process_data import categorical_encode, feature_engineer
from steps.refine_model import remove_insignificant_vars
from steps.train_model import re_train, train

logger = get_logger(__name__)
from zenml.client import Client

print(Client().active_stack.experiment_tracker.get_tracking_uri())

@pipeline(enable_cache=False)
def training_pipeline(): 
    """Pipeline for training."""
    try: 
        logger.info("Training pipeline started")
        # Ingest data
        df = ingest("retail_prices")
        # Process data
        df_processed = categorical_encode(df)
        df_transformed = feature_engineer(df_processed)  
        # Split data 
        X_train, X_test, y_train, y_test = split_data(df_transformed) 
        logger.info(X_train)
        # Train model
        model = train(X_train, y_train) 
        predictors = remove_insignificant_vars(model, df_transformed, alpha=0.05) 
        # Evaluate model
        rmse = evaluate(model, df_transformed)

        # Re-train model
        model1, df_with_significant_vars = re_train(X_train, y_train, predictors)  
        rmse1 = evaluate(model1, df_with_significant_vars)

        logger.info("Training pipeline completed")
    except Exception as e:
        logger.error(e)
        raise e