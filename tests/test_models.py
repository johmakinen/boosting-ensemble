import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(str(Path(os.getcwd()).parent))
from src.models import EnsembleModel
from src.configs import config_test


def classification_data(n_classes=2):
    # Create dummy datasets for testing
    train_data = pd.DataFrame(
        np.random.rand(100, 5),
        columns=["feat1", "feat2", "feat3", "feat4", "feat5"],
    )
    train_target = pd.Series(np.random.randint(0, n_classes, 100))

    val_data = pd.DataFrame(
        np.random.rand(50, 5), columns=["feat1", "feat2", "feat3", "feat4", "feat5"]
    )
    val_target = pd.Series(np.random.randint(0, n_classes, 50))
    test_data = pd.DataFrame(
        np.random.rand(30, 5), columns=["feat1", "feat2", "feat3", "feat4", "feat5"]
    )
    test_target = pd.Series(np.random.randint(0, n_classes, 30))

    return {
        "train": (train_data, train_target),
        "val": (val_data, val_target),
        "test": (test_data, test_target),
    }


def regression_data():
    # Create dummy datasets for testing
    train_data = pd.DataFrame(
        np.random.rand(100, 5),
        columns=["feat1", "feat2", "feat3", "feat4", "feat5"],
    )
    train_target = pd.Series(np.random.rand(100))
    val_data = pd.DataFrame(
        np.random.rand(50, 5), columns=["feat1", "feat2", "feat3", "feat4", "feat5"]
    )
    val_target = pd.Series(np.random.rand(50))
    test_data = pd.DataFrame(
        np.random.rand(30, 5), columns=["feat1", "feat2", "feat3", "feat4", "feat5"]
    )
    test_target = pd.Series(np.random.rand(30))

    return {
        "train": (train_data, train_target),
        "val": (val_data, val_target),
        "test": (test_data, test_target),
    }


@pytest.mark.parametrize(
    "task,curr_data",
    [
        ("classification", classification_data(2)),
        ("classification", classification_data(3)),
        ("regression", regression_data()),
    ],
)
def test_fill_params(task, curr_data):
    config = config_test.copy()
    model = EnsembleModel(config, task, curr_data)
    model.fill_params_()
    assert model.config["params"][task]["xgboost"]["objective"] == (
        "multi:softprob" if task == "classification" else "reg:squarederror"
    )
    assert model.config["params"][task]["lightgbm"]["objective"] == (
        "multiclass" if task == "classification" else "regression"
    )
    assert model.config["params"][task]["catboost"] is not None

    # Datasets are filled?
    assert model.config["train_params"]["xgboost"]["dtrain"] is not None
    assert model.config["train_params"]["lightgbm"]["train_set"] is not None
    assert model.config["train_params"]["catboost"]["pool"] is not None

    assert model.config["train_params"]["xgboost"]["evals"][0][0] is not None
    assert model.config["train_params"]["xgboost"]["evals"][1][0] is not None
    assert None not in model.config["train_params"]["lightgbm"]["valid_sets"]
    assert model.config["train_params"]["catboost"]["eval_set"] is not None


@pytest.mark.parametrize(
    "task,curr_data",
    [
        ("classification", classification_data(2)),
        ("classification", classification_data(3)),
        ("regression", regression_data()),
    ],
)
def test_train_basemodels(task, curr_data):
    config = config_test.copy()
    model = EnsembleModel(config, task, curr_data)
    print(model.config["params"][task])
    model.train_basemodels()
    assert len(model.fitted_models) == 3
    assert model.fitted_models[0].num_boosted_rounds() > 0  # xgboost
    assert model.fitted_models[1].current_iteration() > 0  # lightgbm
    assert model.fitted_models[2].tree_count_ > 0  # catboost


@pytest.mark.parametrize(
    "task,curr_data",
    [
        ("classification", classification_data(2)),
        ("classification", classification_data(3)),
        ("regression", regression_data()),
    ],
)
def test_get_meta_features(task, curr_data):
    config = config_test.copy()
    model = EnsembleModel(config, task, curr_data)
    model.train_basemodels()
    meta_features = model.get_meta_features_()
    assert meta_features["train"].get_data().shape[0] == curr_data["train"][0].shape[0]
    assert meta_features["val"].get_data().shape[0] == curr_data["val"][0].shape[0]
    assert meta_features["test"].get_data().shape[0] == curr_data["test"][0].shape[0]


@pytest.mark.parametrize(
    "task,curr_data",
    [
        ("classification", classification_data(2)),
        ("classification", classification_data(3)),
        ("regression", regression_data()),
    ],
)
def test_train(task, curr_data):
    config = config_test.copy()
    model = EnsembleModel(config, task, curr_data)
    model.train()
    assert model.final_model is not None


if __name__ == "__main__":
    pytest.main()
