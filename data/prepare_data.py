# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914
"""
Process raw dataset for experiments
"""
import os

import argparse
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bias_prob",
        type=float,
        required=False,
        default=0.65,
        help="probability bias_variable=0 if loan is defaulted. \
              probability bias_variable=1 if loan is not defaulted."
    )
    flags = parser.parse_args()

    # Loading the data from this Kaggle link:
    # https://www.kaggle.com/datasets/laotse/credit-risk-dataset.
    data = pl.read_csv(os.path.join("data", "credit_risk_dataset.csv"))

    # synthesizing a biased variable, for fairness evaluation
    rng = np.random.default_rng(seed=0)
    default_bias = rng.choice(2, p=[flags.bias_prob, 1-flags.bias_prob], size=data.shape[0])
    non_default_bias = rng.choice(2, p=[1-flags.bias_prob, flags.bias_prob], size=data.shape[0])

    # bias is conditional on label
    data = (
        data
        .with_columns(
            pl.when(pl.col("loan_status") == 1)
            .then(default_bias)
            .otherwise(non_default_bias)
            .alias("bias_variable")
        )
    )

    # hold out test set for evaluation
    data_train, data_test = train_test_split(data, test_size=0.7, random_state=0)

    # saving the generated files for further usage
    data_train.write_parquet(os.path.join("data", "credit_risk_train.parquet"))
    data_test.write_parquet(os.path.join("data", "credit_risk_test.parquet"))
