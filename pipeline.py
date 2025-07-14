from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from logger import get_logger
from utils import get_data_by_version
from mlflow_client import MLflowClientWrapper
from optimizer import OptunaOptimizer
from trainer import LightGBMTrainer
import mlflow


class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger("pipeline")
        self.client = MLflowClientWrapper(
            config["experiment_name"], config.get("mlflow_uri", None)
        )

    def run(self):
        version = self.config["data_version"]
        df, desc = get_data_by_version(version)

        self.logger.info(f"📦 Training on data version {version}: {desc}")
        X = df.drop(self.config["target_column"], axis=1)
        y = df[self.config["target_column"]]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ✅ Start parent MLflow run
        with self.client.start_run(run_name=f"data_v{version}") as run:
            run_id = run.info.run_id

            opt = OptunaOptimizer(X_train, y_train, X_val, y_val, self.config["n_trials"])
            best_params = opt.optimize()
            self.logger.info(f"🎯 Best hyperparameters found: {best_params}")

            model = LightGBMTrainer(best_params).train(X_train, y_train, X_val, y_val)
            preds = model.predict(X_val)
            auc = roc_auc_score(y_val, preds)
            self.logger.info(f"📈 Validation AUC: {auc:.4f}")

            self.client.log_params(best_params)
            self.client.log_metrics({"val_auc": auc})
            self.client.log_model(model, "model", X_train=X_train)

            mlflow.log_param("data_version", version)
            mlflow.log_param("data_description", desc)
            self.logger.info(f"📌 Logged data_version: {version}, description: {desc}")

            self.client.register_model(run_id, "model", self.config["model_name"])
            self.logger.info(
                f"✅ Training complete for data v{version} — AUC: {auc:.4f}"
            )
