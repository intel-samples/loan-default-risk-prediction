# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Compute predictions from a trained model,
optionally with Intel optimizations
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



def main(flags):
    """Get predictions from a trained model on an input file

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

    with open(flags.saved_model, "rb") as model_file:
        classifier_model = pickle.load(model_file)

    # Load data on which to make predictions
    data = pl.read_csv(flags.input_file)

    # Calculate the loan default (non-payment) probabilities
    if not flags.intel:
        inf_start = time.time()
        preds = classifier_model.predict_proba(data)
        inf_end = time.time()
    else:
        # For the Intel optimizations, a different framework is used.
        # Note: 'daal4py' comes from package 'scikit-learn-intelex'.
        # The XGBoost model first needs to be converted to this framework,
        # and then used as a replacement of the XGBoost component.
        # In practice, one might usually want to save the result from this
        # transformation (through 'pickle') instead of producing it anew
        # for each call.
        import daal4py
        optimized_classifier = daal4py.mb.convert_model(classifier_model.named_steps["xgb"])
        transformer = classifier_model.named_steps["col_transformer"].set_output(transform="default")

        inf_start = time.time()
        features = transformer.transform(data)
        preds = optimized_classifier.predict_proba(features)
        inf_end = time.time()

    if not flags.silent:
        logger.info("First 5 predictions")
        logger.info(preds[:5])
    if flags.intel:
        logger.info("Inference time (Intel-optimized): %f", inf_end - inf_start)
    else:
        logger.info("Inference time (XGBoost): %f", inf_end - inf_start)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--intel",
        action="store_true",
        required=False,
        default=False,
        help="Toggle to use Intel-optimized prediction routines",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        required=False,
        default=False,
        help="Don't print predictions.",
    )
    parser.add_argument(
        "--saved_model",
        type=str,
        required=False,
        default=os.path.join("saved_models", "classifier_model.pkl"),
        help="Saved model file from 'run_training.py'.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=False,
        default=os.path.join("data", "credit_risk_dataset.csv"),
        help="Input file for inference",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="",
        help="Log file to output benchmarking results to.",
    )

    FLAGS = parser.parse_args()
    main(FLAGS)
