"""
This module contains the implementation of the EnsembleModel class.
"""

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
from src.utils import fill_params, evaluate_model

import logging

logger = logging.getLogger(__name__)

# TODO: Add hyperopt
# TODO: Add Dask support
# TODO: Add benchmarks for some non-trivial datasets (e.g., Kaggle/PapersWithCode)

class EnsembleModel:
    """
    Initialize the BoostingEnsemble model.

    Args:
        config (dict): Configuration parameters for the model.
        task (str): The task type, e.g., "classification" or "regression".
        datasets_ (dict[str, tuple[pd.DataFrame, pd.Series]]): Dictionary containing the datasets for training, validation, and testing.

    Attributes:
        config (dict): Configuration parameters for the model.
        task (str): The task type, e.g., "classification" or "regression".
        models (list): List of boosting models to be used.
        fitted_models (list): List to store the fitted models.
        meta_data: Placeholder for storing meta data.
        final_model: Placeholder for storing the final model.
        datasets_ (dict[str, tuple[pd.DataFrame, pd.Series]]): Dictionary containing the datasets for training, validation, and testing.
    """

    def __init__(
        self,
        config: dict,
        task: str,
        datasets_: dict[str, tuple[pd.DataFrame, pd.Series]],
    ):

        self.config = config
        self.task = task
        self.models = [xgb, lgb, ctb]
        self.fitted_models = []
        self.meta_data = None
        self.final_model = None
        self.datasets_ = datasets_
        self.eval_results = None

        self.fill_params_()

    def fill_params_(self):
        """
        Fills the parameters for the model based on the training and validation datasets.
        """
        self.config = fill_params(
            *self.datasets_["train"],
            *self.datasets_["val"],
            config=self.config,
            task=self.task,
        )

    def train_basemodels(self):
        """
        Trains the base models.

        This method iterates over the list of models and trains each model using the specified parameters
        from the configuration file. The trained models are then appended to the `fitted_models` list.
        """
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
        """
        Get meta features from the fitted models. (Their predictions on the train set)

        Raises:
            ValueError: If the base models are not trained yet.

        Returns:
            dict: A dictionary containing the meta features for the train, val, and test sets.
        """
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
                    str(x.__module__).split(".", maxsplit=1)[0]
                    for x in self.fitted_models
                ]
                for i in range(n_cols)
            ]
            res[set_] = xgb.DMatrix(x_meta, y_)
            logger.info("%s meta features created", set_)
        return res

    def train(self):
        """
        Trains the ensemble model.

        This method trains the ensemble model by first training the base models,
        then obtaining the meta features, and finally training the final model.
        """
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

    def evaluate(self,plot=True):
        """
        Evaluates the final model of the ensemble.

        Raises:
            ValueError: If the final model is not trained yet.
        """
        if self.final_model is None:
            raise ValueError("Train ensemble first")
        self.eval_results = evaluate_model(
            model=self.final_model,
            config=self.config,
            meta_data=self.meta_data,
            task=self.task,
            plot=plot,
        )
        logger.info("Model evaluation complete. Results saved in model.eval_results")

