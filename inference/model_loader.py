import mlflow
from mlflow.tracking import MlflowClient
from logger import get_logger
import yaml

logger = get_logger("model_loader")


def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


config = load_config()


def load_best_model():
    client = MlflowClient()
    model_name = config["model_name"]
    versions = client.search_model_versions(f"name='{model_name}'")

    best_model_uri = None
    best_val_auc = -1.0
    best_version = None

    for v in versions:
        run_id = v.run_id
        run = client.get_run(run_id)
        val_auc = run.data.metrics.get("val_auc")

        if val_auc is not None:
            logger.info(f"Found model version {v.version} with val_auc: {val_auc}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_uri = f"runs:/{run_id}/model"
                best_version = v.version

    if not best_model_uri:
        logger.error("❌ No model with 'val_auc' metric found.")
        raise ValueError("No model with val_auc metric found.")

    logger.info(f"✅ Loading model version {best_version} from URI: {best_model_uri} with val_auc: {best_val_auc}")
    model = mlflow.pyfunc.load_model(best_model_uri)

    try:
        input_example = model.metadata.get_input_schema()
        column_names = [col.name for col in input_example]
        logger.info(f"Model version {best_version} expects input columns (in order): {column_names}")
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch model input signature: {e}")
        column_names = []

    model.expected_columns = column_names
    return model
