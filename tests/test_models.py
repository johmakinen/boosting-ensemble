import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(str(Path(os.getcwd()).parent))
from src.models import EnsembleModel
from src.configs import config_test


class TestEnsembleModel(unittest.TestCase):
    def setUp(self):
        # Create dummy datasets for testing
        train_data = pd.DataFrame(
            np.random.rand(100, 5),
            columns=["feat1", "feat2", "feat3", "feat4", "feat5"],
        )
        train_target = pd.Series(np.random.randint(0, 2, 100))
        val_data = pd.DataFrame(
            np.random.rand(50, 5), columns=["feat1", "feat2", "feat3", "feat4", "feat5"]
        )
        val_target = pd.Series(np.random.randint(0, 2, 50))
        test_data = pd.DataFrame(
            np.random.rand(30, 5), columns=["feat1", "feat2", "feat3", "feat4", "feat5"]
        )
        test_target = pd.Series(np.random.randint(0, 2, 30))

        self.datasets = {
            "train": (train_data, train_target),
            "val": (val_data, val_target),
            "test": (test_data, test_target),
        }

        self.config = config_test

        self.model = EnsembleModel(self.config, "classification", self.datasets)

    def test_fill_params(self):
        self.model.fill_params_()
        self.assertEqual(
            self.model.config["params"]["classification"]["xgboost"]["objective"],
            "multi:softprob",
        )
        self.assertEqual(
            self.model.config["params"]["classification"]["lightgbm"]["objective"],
            "multiclass",
        )
        self.assertEqual(
            self.model.config["params"]["classification"]["catboost"]["loss_function"],
            "Logloss",
        )

    def test_train_basemodels(self):
        self.model.train_basemodels()
        self.assertEqual(len(self.model.fitted_models), 3)

    def test_get_meta_features(self):
        self.model.train_basemodels()
        meta_features = self.model.get_meta_features_()
        self.assertEqual(
            meta_features["train"].get_data().shape[0],
            self.datasets["train"][0].shape[0],
        )
        self.assertEqual(
            meta_features["val"].get_data().shape[0], self.datasets["val"][0].shape[0]
        )
        self.assertEqual(
            meta_features["test"].get_data().shape[0], self.datasets["test"][0].shape[0]
        )

    def test_train(self):
        self.model.train()
        self.assertIsNotNone(self.model.final_model)


if __name__ == "__main__":
    unittest.main()

#  python -m unittest -v
