"""
This file contains the configuration for the training process of the models.
"""

import lightgbm as lgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval
from hyperopt.early_stop import no_progress_loss
import numpy as np

config = {
    "train_params": {
        "xgboost": {
            "dtrain": None,
            "evals": [(None, "train"), (None, "val")],
            "num_boost_round": 300,
            "early_stopping_rounds": 20,
            "verbose_eval": 0,
        },
        "lightgbm": {
            "train_set": None,
            "valid_sets": [None, None],
            "num_boost_round": 300,
            "callbacks": [
                lgb.early_stopping(
                    stopping_rounds=20,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        },
        "catboost": {
            "pool": None,
            "eval_set": None,
            "num_boost_round": 300,
            "early_stopping_rounds": 20,
            "verbose": 0,
        },
        "final_model": {
            "dtrain": None,
            "evals": [(None, "train"), (None, "val")],
            "num_boost_round": 300,
            "early_stopping_rounds": 20,
            "verbose_eval": 0,
        },
    },
    "params": {
        "classification": {
            "xgboost": {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": None,
            },
            "lightgbm": {
                "objective": "multiclass",
                "metric": "multi_logloss",
                "num_class": None,
                "verbose": -1,
            },
            "catboost": {
                "loss_function": "Logloss",
                "classes_count": None,
            },
            "final_model": {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": None,
            },
        },
        "regression": {
            "xgboost": {
                "objective": "reg:squarederror",
            },
            "lightgbm": {
                "objective": "regression",
                "verbose": -1,
            },
            "catboost": {
                "loss_function": "RMSE",
            },
            "final_model": {
                "objective": "reg:squarederror",
            },
        },
    },
}

config_test = {
    "train_params": {
        "xgboost": {
            "dtrain": None,
            "evals": [(None, "train"), (None, "val")],
            "num_boost_round": 10,
            "early_stopping_rounds": 2,
            "verbose_eval": 0,
        },
        "lightgbm": {
            "train_set": None,
            "valid_sets": [None, None],
            "num_boost_round": 10,
            "callbacks": [
                lgb.early_stopping(
                    stopping_rounds=2,
                    verbose=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        },
        "catboost": {
            "pool": None,
            "eval_set": None,
            "num_boost_round": 10,
            "early_stopping_rounds": 2,
            "verbose": 0,
        },
        "final_model": {
            "dtrain": None,
            "evals": [(None, "train"), (None, "val")],
            "num_boost_round": 10,
            "early_stopping_rounds": 2,
            "verbose_eval": 0,
        },
    },
    "params": {
        "classification": {
            "xgboost": {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": None,
            },
            "lightgbm": {
                "objective": "multiclass",
                "metric": "multi_logloss",
                "num_class": None,
                "verbose": -1,
            },
            "catboost": {
                "loss_function": "Logloss",
                "classes_count": None,
            },
            "final_model": {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": None,
            },
        },
        "regression": {
            "xgboost": {
                "objective": "reg:squarederror",
            },
            "lightgbm": {
                "objective": "regression",
                "verbose": -1,
            },
            "catboost": {
                "loss_function": "RMSE",
            },
            "final_model": {
                "objective": "reg:squarederror",
            },
        },
    },
}


hpo_spaces = {
    "xgboost": {
        "max_depth": hp.choice("max_depth", np.arange(1, 20, 1, dtype=int)),
        "eta": hp.uniform("eta", 0, 1),
        "gamma": hp.uniform("gamma", 0, 10e1),
        "reg_alpha": hp.uniform("reg_alpha", 10e-7, 10),
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "colsample_bynode": hp.uniform("colsample_bynode", 0.5, 1),
        "colsample_bylevel": hp.uniform("colsample_bylevel", 0.5, 1),
        "min_child_weight": hp.choice(
            "min_child_weight", np.arange(1, 10, 1, dtype="int")
        ),
        "max_delta_step": hp.choice("max_delta_step", np.arange(1, 10, 1, dtype="int")),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "seed": 42,
    },
    "lightgbm": {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
        "num_leaves": hp.choice("num_leaves", np.arange(2, 100, 1, dtype=int)),
        "max_depth": hp.choice("max_depth", np.arange(1, 20, 1, dtype=int)),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
        "reg_alpha": hp.uniform("reg_alpha", 0, 1),
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "n_jobs": 2,
        "random_state": 42,
        "verbose": -1,
        "verbose_eval": -1,
    },
    "catboost": {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
        "depth": hp.choice("depth", np.arange(1, 16, 1, dtype=int)),
        "subsample": hp.uniform("subsample", 0.5, 1),
        "colsample_bylevel": hp.uniform("colsample_bylevel", 0.5, 1),
        "l2_leaf_reg": hp.uniform("l2_leaf_reg", 0, 1),
        "random_state": 42,
        "verbose": 0,
    },
}
