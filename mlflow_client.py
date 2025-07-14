import mlflow
from mlflow.models.signature import infer_signature
import mlflow.lightgbm
from logger import get_logger

logger = get_logger("mlflow_client")

class MLflowClientWrapper:
    def __init__(self, experiment_name, tracking_uri=None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI set to {tracking_uri}")
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set: {experiment_name}")

    def start_run(self, run_name=None):
        logger.info(f"Starting new MLflow run: {run_name}")
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params):
        logger.info(f"Logging parameters: {params}")
        mlflow.log_params(params)

    def log_metrics(self, metrics):
        logger.info(f"Logging metrics: {metrics}")
        mlflow.log_metrics(metrics)

    def log_model(self, model, artifact_path, X_train=None):
        """
        Logs the LightGBM model with inferred input signature and example
        if X_train is provided.
        """
        try:
            signature = None
            input_example = None

            if X_train is not None:
                # Convert int columns to float to avoid schema mismatch at inference
                X_train = X_train.astype({col: 'float64' for col in X_train.columns if X_train[col].dtype == 'int64'})
                signature = infer_signature(X_train, model.predict(X_train))
                input_example = X_train.head(1)
                logger.info("Inferred input signature and prepared input example.")

            logger.info(f"Logging model to MLflow at path: {artifact_path}")
            mlflow.lightgbm.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example
            )
            logger.info("✅ Model logged successfully.")

        except Exception as e:
            logger.error(f"❌ Error while logging model: {e}")
            raise

    def register_model(self, run_id, artifact_path, model_name):
        try:
            uri = f"runs:/{run_id}/{artifact_path}"
            logger.info(f"Registering model from URI: {uri} as '{model_name}'")
            mlflow.register_model(uri, model_name)
            logger.info("✅ Model registered successfully.")
        except Exception as e:
            logger.error(f"❌ Model registration failed: {e}")
            raise
