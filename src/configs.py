"""
This file contains the configuration for the training process of the models.
"""

import lightgbm as lgb

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
