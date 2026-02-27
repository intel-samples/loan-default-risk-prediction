# !/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Fits an XGBoost model to credit bureau data, transforming
features as necessary in a scikit-learn pipeline
"""

import argparse
import pathlib
import logging
import time
import warnings
import pickle
import os

import numpy as np
import polars as pl
import xgboost as xgb

from utils.fairness import get_fairness_parity_report


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score


def main(flags):
    """Benchmark incremental training with bias info

    Args:
        flags: training configuration
    """

    if flags.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        path = pathlib.Path(flags.logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=flags.logfile, level=logging.DEBUG)
    logger = logging.getLogger()

    # load data
    data_train = pl.read_parquet(flags.train_file)

    # Don't train on bias variable
    X_train = data_train.drop(["loan_status", "bias_variable"])
    y_train = data_train["loan_status"]

    # define model using scikit-learn idioms
    classifier_model = Pipeline(
        steps=[
            (
                "col_transformer",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            SimpleImputer(strategy="median"),
                            ["loan_int_rate", "person_emp_length", "cb_person_cred_hist_length"],
                        ),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=5),
                            ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
                        ),
                    ],
                    remainder="passthrough",
                )
            ),
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.01,
                    max_depth=10,
                    min_child_weight=6,
                    verbosity=0,
                ),
            ),
        ]
    ).set_output(transform="polars")

    start = time.time()
    clf = classifier_model.fit(X_train, y_train)
    end = time.time()
    logger.info("Model fitting time : %f seconds", end-start)

    # Evaluate model performance metrics on hold out test set
    data_test = pl.read_parquet(flags.test_file)
    X_test = data_test.drop(["loan_status", "bias_variable"])
    y_test = data_test["loan_status"]

    predicted_class = classifier_model.predict(X_test)
    predicted_probabilities = classifier_model.predict_proba(X_test)
    logger.info(classification_report(
        predicted_class, y_test
    ))
    auc = roc_auc_score(y_test, predicted_probabilities[:, 1])
    logger.info("AUROC : %f", auc)

    # record fairness metrics for given model on holdout test set
    parity_values = get_fairness_parity_report(
        classifier_model,
        X_test, y_test,
        data_test["bias_variable"],
    )

    print("Parity Ratios (Privileged/Non-Privileged):")
    for k, v in parity_values.items():
        print(f"\t{k.upper()} : {v:.2f}")

    # save model and preprocessor as a unified pipeline
    if flags.save_model_path is not None:
        path = pathlib.Path(flags.save_model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path / "classifier_model.pkl", "wb") as out_file:
            pickle.dump(classifier_model, out_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_model_path",
        type=str,
        required=False,
        default="saved_models",
        help="Path to save the fitted model.  If not provided, does not save."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=False,
        default=os.path.join("data", "credit_risk_train.parquet"),
        help="Data file for training (parquet format)."
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=False,
        default=os.path.join("data", "credit_risk_test.parquet"),
        help="Data file for testing (parquet format)."
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="",
        help="Log file to output benchmarking results to.")

    FLAGS = parser.parse_args()
    main(FLAGS)
