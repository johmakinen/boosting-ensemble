import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from src.utils import fill_params, evaluate_model

import logging
logger = logging.getLogger(__name__)

# TODO: Add docstrings
# TODO: Add hyperopt

class EnsembleModel:
    def __init__(self, config, task, datasets_):
        self.config = config
        self.task = task
        self.models = [xgb, lgb, ctb]
        self.fitted_models = []
        self.meta_data = None
        self.final_model = None
        self.datasets_ = datasets_  # {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
        self.fill_params_()

    def fill_params_(self):
        self.config = fill_params(
            *self.datasets_["train"],
            *self.datasets_["val"],
            config=self.config,
            task=self.task,
        )

    def train_basemodels(self):
        logger.info("Training base models")
        for class_ in self.models:
            trainer = getattr(class_, "train")
            model = trainer(
                params=self.config["params"][self.task][class_.__name__.lower()],
                **self.config["train_params"][class_.__name__.lower()],
            )
            self.fitted_models.append(model)
            logger.info("%s trained", class_.__name__)

    def get_meta_features_(self):
        if self.fitted_models is None:
            raise ValueError("Train base models first")
        logger.info("Getting meta features")
        res = {"train": None, "val": None, "test": None}
        xgb_model, lgb_model, ctb_model = self.fitted_models
        for set_ in self.datasets_:
            x_, y_ = self.datasets_[set_]
            xgb_preds = xgb_model.predict(xgb.DMatrix(x_))
            lgb_preds = lgb_model.predict(x_)
            ctb_preds = ctb_model.predict(
                x_,
                prediction_type=(
                    "Probability" if self.task == "classification" else "RawFormulaVal"
                ),
            )

            # For now, no predict probas
            x_meta = pd.concat(
                [
                    pd.DataFrame(xgb_preds),
                    pd.DataFrame(lgb_preds),
                    pd.DataFrame(ctb_preds),
                ],
                axis=1,
            )

            n_cols = (
                1
                if self.task == "regression"
                else self.config["params"][self.task]["final_model"]["num_class"]
            )
            x_meta.columns = [
                f"{_name}{i}"
                for _name in [
                    str(x.__module__).split(".")[0] for x in self.fitted_models
                ]
                for i in range(n_cols)
            ]
            res[set_] = xgb.DMatrix(x_meta, y_)
            logger.info("%s meta features created", set_)
        return res

    def train(self):
        logger.info("Training ensemble")
        self.train_basemodels()
        self.meta_data = self.get_meta_features_()

        self.config["train_params"]["final_model"]["dtrain"] = self.meta_data["train"]
        self.config["train_params"]["final_model"]["evals"] = [
            (self.meta_data["train"], "train"),
            (self.meta_data["val"], "val"),
        ]

        self.final_model = xgb.train(
            params=self.config["params"][self.task]["final_model"],
            **self.config["train_params"]["final_model"],
        )
        logger.info("Ensemble trained")

    def evaluate(self):
        if self.final_model is None:
            raise ValueError("Train ensemble first")
        evaluate_model(model=self.final_model, config=self.config,meta_data=self.meta_data, task=self.task)