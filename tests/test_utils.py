import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb

sys.path.append(str(Path(os.getcwd()).parent))
from src.utils import fill_params


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Create dummy datasets for testing
        self.X_train = pd.DataFrame(
            np.random.rand(100, 5),
            columns=["feat1", "feat2", "feat3", "feat4", "feat5"],
        )
        self.y_train = pd.Series(np.random.randint(0, 2, 100))
        self.X_val = pd.DataFrame(
            np.random.rand(50, 5), columns=["feat1", "feat2", "feat3", "feat4", "feat5"]
        )
        self.y_val = pd.Series(np.random.randint(0, 2, 50))

        self.config = {
            "train_params": {"xgboost": {}, "lightgbm": {}, "catboost": {}},
            "params": {
                "classification": {
                    "xgboost": {},
                    "lightgbm": {},
                    "catboost": {},
                    "final_model": {},
                }
            },
        }

        self.task = "classification"

        self.updated_config = fill_params(
            self.X_train, self.y_train, self.X_val, self.y_val, self.config, self.task
        )

    def test_fill_params_xgboost(self):
        self.assertIsInstance(
            self.updated_config["train_params"]["xgboost"]["dtrain"], xgb.DMatrix
        )
        self.assertEqual(
            len(self.updated_config["train_params"]["xgboost"]["evals"]), 2
        )
        self.assertIsInstance(
            self.updated_config["train_params"]["xgboost"]["evals"][0][0], xgb.DMatrix
        )
        self.assertIsInstance(
            self.updated_config["train_params"]["xgboost"]["evals"][1][0], xgb.DMatrix
        )

    def test_fill_params_lightgbm(self):
        self.assertIsInstance(
            self.updated_config["train_params"]["lightgbm"]["train_set"], lgb.Dataset
        )
        self.assertEqual(
            len(self.updated_config["train_params"]["lightgbm"]["valid_sets"]), 2
        )
        self.assertIsInstance(
            self.updated_config["train_params"]["lightgbm"]["valid_sets"][0],
            lgb.Dataset,
        )
        self.assertIsInstance(
            self.updated_config["train_params"]["lightgbm"]["valid_sets"][1],
            lgb.Dataset,
        )

    def test_fill_params_catboost(self):
        self.assertIsInstance(
            self.updated_config["train_params"]["catboost"]["pool"], ctb.Pool
        )
        self.assertIsInstance(
            self.updated_config["train_params"]["catboost"]["eval_set"], ctb.Pool
        )

    def test_fill_params_classification(self):
        self.assertEqual(
            self.updated_config["params"]["classification"]["xgboost"]["num_class"],
            len(set(self.y_train) & set(self.y_val)),
        )
        self.assertEqual(
            self.updated_config["params"]["classification"]["lightgbm"]["num_class"],
            len(set(self.y_train) & set(self.y_val)),
        )
        self.assertEqual(
            self.updated_config["params"]["classification"]["final_model"]["num_class"],
            len(set(self.y_train) & set(self.y_val)),
        )


if __name__ == "__main__":
    unittest.main()


# python -m unittest -v
