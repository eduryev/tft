# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Trains TFT based on a defined set of parameters.

Uses default parameters supplied from the configs file to train a TFT model from
scratch.

Usage:
python3 script_train_fixed_params {expt_name} {output_folder}

Command line args:
  expt_name: Name of dataset/experiment to train.
  output_folder: Root folder in which experiment is saved


"""

import argparse
import datetime as dte
import os

import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

import dask.dataframe as dd
from time import time

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer
tf.experimental.output_all_intermediates(True)


def main(expt_name,
         use_gpu,
         model_folder,
         data_csv_path,
         data_formatter,
         testing_mode_params=None):
    """Trains tft based on defined model params.

    Args:
      expt_name: Name of experiment
      use_gpu: Whether to run tensorflow with GPU operations
      model_folder: Folder path where models are serialized
      data_csv_path: Path to csv file containing data
      data_formatter: Dataset-specific data fromatter (see
        expt_settings.dataformatter.GenericDataFormatter)
      testing_mode_params: Uses a smaller models and data sizes for testing purposes
        only -- switch to False to use original default settings
    """

    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError(
            "Data formatters should inherit from" +
            "AbstractDataFormatter! Type={}".format(type(data_formatter)))

    # Tensorflow setup
    default_keras_session = tf.compat.v1.keras.backend.get_session()

    if use_gpu:
        tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=0)

    else:
        tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

    print("*** Training from defined parameters for {} ***".format(expt_name))

    print("** Loading & splitting data... **")
    start_time = time()
    raw_data = dd.read_csv(data_csv_path)
    raw_data = raw_data.compute()
    raw_data.sort_index(inplace=True)
    end_time = time()
    print(f"Loaded raw data in {end_time - start_time:.2f}s")
    # with pd.option_context('mode.use_inf_as_null', True):
    #     raw_data = raw_data.dropna()
    raw_data.replace({'week_of_year': {53: 52}}, inplace=True)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()
    print("** Data loading & splitting is finished. **")

    id_to_weight = raw_data.groupby('Asset_ID')['Weight'].mean().to_dict()

    print("** Setting parameters... **")
    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Parameter overrides for testing only! Small sizes used to speed up script.
    if testing_mode_params:
        for param in fixed_params:
            if param in testing_mode_params:
                fixed_params[param] = testing_mode_params[param]
        for param in params:
            if param in testing_mode_params:
                params[param] = testing_mode_params[param]
        if "train_samples" in testing_mode_params:
            train_samples = testing_mode_params["train_samples"]
        if "valid_samples" in testing_mode_params:
            valid_samples = testing_mode_params["valid_samples"]
    print("** Parameters have been set. **")

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)

    # Training -- one iteration only
    print("*** Running calibration ***")
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

        tf.keras.backend.set_session(sess)

        print("*** Running training ***")
        params = opt_manager.get_next_parameters()
        model = ModelClass(params, use_cudnn=use_gpu)

        if not model.training_data_cached():
            model.cache_batched_data(train, "train", num_samples=train_samples)
            model.cache_batched_data(valid, "valid", num_samples=valid_samples)

        sess.run(tf.global_variables_initializer())
        model.fit()

        print("*** Running validation ***")
        val_loss = model.evaluate()

        print("*** Running tests ***")

        print("** Generating model predictions... **")
        output_map = model.predict(test, return_targets=True)
        print("** Model predictions are generated. **")

        print("** Formatting predictions... **")
        targets = data_formatter.format_predictions(output_map["targets"])
        p50_forecast = data_formatter.format_predictions(output_map["p50"])
        p90_forecast = data_formatter.format_predictions(output_map["p90"])
        print("** Predictions formatting is finished. **")

        weights = np.vectorize(id_to_weight.get)(targets.identifier.values)

        def weighted_mean(y, weights):
            return np.sum(weights * y) / np.sum(weights)

        def weighted_cov(x, y, weights):
            x_w_mean = weighted_mean(x, weights)
            y_w_mean = weighted_mean(y, weights)
            return np.sum(weights * (x - x_w_mean) * (y - y_w_mean)) / np.sum(weights)

        def weighted_corr(x, y, weights):
            return weighted_cov(x, y, weights) / np.sqrt(weighted_cov(x, x, weights) * weighted_cov(y, y, weights))


        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        correlation_score = weighted_corr(targets['t+0'].values, p50_forecast['t+0'].values, weights)

        p50_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(p50_forecast),
            0.5)
        p90_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(p90_forecast),
            0.9)

        tf.keras.backend.set_session(default_keras_session)

    results_path = 'output/results/crypto/results.csv'
    results_dict = params.copy()
    del results_dict['column_definition']
    results_dict["val_loss"] = val_loss
    results_dict["correlation_score"] = correlation_score
    results_dict["p50_loss"] = p50_loss
    results_dict["p90_loss"] = p90_loss
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame(columns=list(results_dict.keys()))
    results_df = results_df.append(results_dict, ignore_index=True)
    results_df.to_csv(results_path)

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    # print("Params:")
    # for k in best_params:
    #     print(k, " = ", best_params[k])
    print()
    print("Normalised Quantile Loss for Test Data: P50={}, P90={}".format(
        p50_loss.mean(), p90_loss.mean()))
    print("Weighted correlation score = {}".format(correlation_score))


if __name__ == "__main__":
    def get_args():
        """Gets settings from command line."""

        experiment_names = ExperimentConfig.default_experiments

        parser = argparse.ArgumentParser(description="Data download configs")
        parser.add_argument(
            "expt_name",
            metavar="e",
            type=str,
            nargs="?",
            default="volatility",
            choices=experiment_names,
            help="Experiment Name. Default={}".format(",".join(experiment_names)))
        parser.add_argument(
            "output_folder",
            metavar="f",
            type=str,
            nargs="?",
            default=".",
            help="Path to folder for data download")
        parser.add_argument(
            "use_gpu",
            metavar="g",
            type=str,
            nargs="?",
            choices=["yes", "no"],
            default="no",
            help="Whether to use gpu for training.")

        args = parser.parse_known_args()[0]

        root_folder = None if args.output_folder == "." else args.output_folder

        return args.expt_name, root_folder, args.use_gpu == "yes"


    name, output_folder, use_tensorflow_with_gpu = get_args()

    print("Using output folder {}".format(output_folder))

    config = ExperimentConfig(name, output_folder)
    formatter = config.make_data_formatter()

    # Customise inputs to main() for new datasets.
    main(
        expt_name=name,
        use_gpu=use_tensorflow_with_gpu,
        model_folder=os.path.join(config.model_folder, "fixed"),
        data_csv_path=config.data_csv_path,
        data_formatter=formatter,
        testing_mode_params=True)  # Change to false to use original default params
