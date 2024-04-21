import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(str(Path(os.getcwd()).parent))
from src.models import EnsembleModel
from src.configs import config_test, hpo_config_test
from src.utils import classification_data, regression_data


@pytest.mark.parametrize(
    "task,curr_data",
    [
        ("classification", classification_data(2)),
        ("classification", classification_data(3)),
        ("regression", regression_data()),
    ],
    ids=["binary_clf", "multi_clf", "reg"],
)
def test_fill_params(task, curr_data):
    config = config_test.copy()
    model = EnsembleModel(config, task, curr_data)
    model.fill_params_()

    if task == "classification":
        n_classes = model.config["params"][task]["xgboost"]["num_class"]
        if n_classes == 2:
            assert (
                model.config["params"][task]["catboost"]["loss_function"] == "Logloss"
            )
            assert "classes_count" not in model.config["params"][task]["catboost"]
        else:
            assert (
                model.config["params"][task]["catboost"]["loss_function"]
                == "MultiClass"
            )
            assert (
                model.config["params"][task]["catboost"]["classes_count"] == n_classes
            )
    else:
        assert "classes_count" not in model.config["params"][task]["catboost"]

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
    ids=["binary_clf", "multi_clf", "reg"],
)
def test_train_basemodels(task, curr_data):
    config = config_test.copy()
    model = EnsembleModel(config, task, curr_data)
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
    ids=["binary_clf", "multi_clf", "reg"],
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
    ids=["binary_clf", "multi_clf", "reg"],
)
def test_train(task, curr_data):
    config = config_test.copy()
    model = EnsembleModel(config, task, curr_data)
    model.train()
    assert model.final_model is not None


@pytest.mark.parametrize(
    "task,curr_data",
    [
        ("classification", classification_data(2)),
        ("classification", classification_data(3)),
        ("regression", regression_data()),
    ],
    ids=["binary_clf", "multi_clf", "reg"],
)
def test_train_hpo(task, curr_data):
    config = config_test.copy()
    hpo_config = hpo_config_test.copy()
    model = EnsembleModel(config, task, curr_data, hpo_config)
    model.train(hpo=True)
    assert model.final_model is not None


if __name__ == "__main__":
    pytest.main()
