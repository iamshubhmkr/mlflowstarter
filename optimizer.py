import optuna
import mlflow
from trainer import LightGBMTrainer
from sklearn.metrics import roc_auc_score


class OptunaOptimizer:
    def __init__(self, X_train, y_train, X_val, y_val, n_trials):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_trials = n_trials

    def objective(self, trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
        }

        # ✅ Start new MLflow run for EACH trial
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            model = LightGBMTrainer(params).train(self.X_train, self.y_train, self.X_val, self.y_val)
            preds = model.predict(self.X_val)
            auc = roc_auc_score(self.y_val, preds)
            mlflow.log_metric("val_auc", auc)

        return auc

    def optimize(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        return study.best_params
