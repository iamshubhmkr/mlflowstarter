import lightgbm as lgb

class LightGBMTrainer:
    def __init__(self, params):
        self.params = params
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)

        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=30)]
        )
        return self.model
