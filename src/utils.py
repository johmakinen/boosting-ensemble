"""
This module contains utility functions for the model training and evaluation.
"""

from sklearn.metrics import (
    RocCurveDisplay,
    auc,
    roc_curve,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
    median_absolute_error,
    r2_score,
)

import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
import logging
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_model(
    model, config: dict, meta_data: dict[xgb.DMatrix], task: str, plot=True
):
    """
    Evaluate the performance of a model on different datasets.

    Args:
        model: The trained model to evaluate.
        config (dict): The configuration parameters for the model.
        meta_data (dict[xgb.DMatrix]): A dictionary containing the datasets to evaluate the model on.
        task (str): The type of task, either "classification" or "regression".
        plot (bool): Whether to plot the evaluation results. Defaults to True.

    Returns:
        None
    """

    logger.info("Evaluating model")
    results = {set_: {} for set_ in meta_data}

    if task == "classification":
        if plot:
            n_classes = config["params"][task]["xgboost"]["num_class"]
            plot_auc = n_classes == 2
            _, ax = plt.subplots(1 if not plot_auc else 2, 3, figsize=(15, 10))
            axes_ = ax.flatten()
        for i, set_ in enumerate(meta_data):
            dmat = meta_data[set_]
            y_ = dmat.get_label()
            y_pred_probs = model.predict(dmat)

            # Compute and add metrics to results.
            # TODO: Add more metrics
            # TODO: Add option for weighted metrics

            for metric in [f1_score, accuracy_score, recall_score, precision_score]:
                results[set_][metric.__name__] = metric(
                    y_, np.argmax(y_pred_probs, axis=1)
                )

            if plot:
                ConfusionMatrixDisplay(
                    confusion_matrix(y_, np.argmax(y_pred_probs, axis=1))
                ).plot(ax=axes_[i], cmap="Blues")
                axes_[i].set_title(f"{set_} set")

                if plot_auc:
                    fpr, tpr, _ = roc_curve(y_, np.argmax(y_pred_probs, axis=1))
                    roc_auc = auc(fpr, tpr)
                    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(
                        ax=axes_[i + 3]
                    )
                    # plot dummy classifier line
                    axes_[i + 3].plot(
                        [0, 1],
                        [0, 1],
                        linestyle="--",
                        lw=2,
                        color="orange",
                        label="Random classifier",
                        alpha=0.8,
                    )
    elif task == "regression":
        if plot:
            _, ax = plt.subplots(4, 3, figsize=(15, 10))
            axes_ = ax.flatten()
        for i, set_ in enumerate(meta_data):
            dmat = meta_data[set_]
            y_ = dmat.get_label()
            y_pred = model.predict(dmat)

            # Compute and add metrics to results.
            for metric in [
                mean_absolute_error,
                mean_absolute_percentage_error,
                mean_squared_error,
                root_mean_squared_error,
                median_absolute_error,
                r2_score,
                mean_squared_error
            ]:
                results[set_][metric.__name__] = metric(y_, y_pred)
            if plot:
                axes_[i].set_title(f"{set_} set")
                # Real vs predicted scatter + "perfect" line which is y=x
                sns.scatterplot(x=y_, y=y_pred, ax=axes_[i])
                axes_[i].set_xlabel("Real")
                axes_[i].set_ylabel("Predicted")
                axes_[i].plot(y_, y_, "--", c="red", alpha=0.5)
                axes_[i + 3].plot(y_pred)
                axes_[i + 3].set_xlabel("Index")
                axes_[i + 3].set_ylabel("Predicted")
                axes_[i + 6].plot(y_pred - y_)
                axes_[i + 6].set_xlabel("Index")
                axes_[i + 6].set_ylabel("Residual (Predicted - Real)")
                # Line at zero
                axes_[i + 6].axhline(0, c="red", linestyle="--", alpha=0.5)
                sns.histplot(y_pred - y_, ax=axes_[i + 9], bins=50, stat="probability")
                axes_[i + 9].set_xlabel("Residual (Predicted - Real)")
    else:
        logger.info("What are you doing?")
    if plot:
        plt.tight_layout()
        plt.show()
    return results


def fill_params(
    X_train, y_train, X_val, y_val, config: dict, task: str = "classification"
):
    """
    Fills the model parameters in the given configuration dictionary.

    Args:
        X_train (array-like): The training input samples.
        y_train (array-like): The target values for the training samples.
        X_val (array-like): The validation input samples.
        y_val (array-like): The target values for the validation samples.
        config (dict): The configuration dictionary to be filled. Defaults to None.
        task (str, optional): The task type. Defaults to "classification".

    Returns:
        dict: The updated configuration dictionary.
    """
    logger.info("Filling model parameters")
    config["train_params"]["xgboost"]["dtrain"] = xgb.DMatrix(X_train, y_train)
    config["train_params"]["xgboost"]["evals"] = [
        (config["train_params"]["xgboost"]["dtrain"], "train"),
        (xgb.DMatrix(X_val, y_val), "val"),
    ]
    config["train_params"]["lightgbm"]["train_set"] = lgb.Dataset(
        X_train, label=y_train
    )
    config["train_params"]["lightgbm"]["valid_sets"] = [
        config["train_params"]["lightgbm"]["train_set"],
        lgb.Dataset(X_val, label=y_val),
    ]
    config["train_params"]["catboost"]["pool"] = ctb.Pool(X_train, label=y_train)
    config["train_params"]["catboost"]["eval_set"] = ctb.Pool(X_val, label=y_val)

    if task == "classification":
        n_classes = len(set(y_train) & set(y_val))
        config["params"][task]["xgboost"]["num_class"] = n_classes
        config["params"][task]["lightgbm"]["num_class"] = n_classes
        config["params"][task]["final_model"]["num_class"] = n_classes
    logger.info("Model parameters filled")
    return config
