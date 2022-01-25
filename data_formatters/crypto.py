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
"""Custom formatting functions for Crypto dataset.

Defines dataset specific column definitions and data transformations.
"""

import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class CryptoFormatter(GenericDataFormatter):
  """Defines and formats data for the crypto dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      ('Asset_ID', DataTypes.CATEGORICAL, InputTypes.ID),
      ('date', DataTypes.DATE, InputTypes.TIME),
      ('Target', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('Count', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('Open', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('High', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('Low', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('Close', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('Volume', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('VWAP', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('log_return', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('Weight', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('days_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('hour_of_day', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('minute_of_day', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('minute_of_hour', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('day_of_month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('week_of_year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('year', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
  ]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None

  def split_data(self,
                 df,
                 start_boundary=0,
                 valid_boundary=1000,
                 test_boundary=1150,
                 end_boundary=1358):
    """Splits data frame into training-validation-test data frames.
    
    There are 1358 days of data in total.
    1000 of them is used for train,
    150 for validation,
    and the rest 208 for test.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')

    index = df['days_from_start']
    train = df.loc[(index >= start_boundary) & (index < valid_boundary)]
    valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
    test = df.loc[(index >= test_boundary) & (index < end_boundary)]

    self.set_scalers(train)

    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    # Extract identifiers in case required
    self.identifiers = list(df[id_column].unique())

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    data = df[real_inputs].values
    self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
    print(f"Target scaler is fit on dataframe shape: {df[[target_column]].values.shape}")
    self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
        df[[target_column]].values)  # used for predictions
    print(f"Target scaler fitting is finished.")

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      print(f"Categorical scaler is fit for column: {col}")
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      print(f"Categorical scaler fitting is finished.")
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()

    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Format real inputs
    output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

    # Format categorical inputs
    for col in categorical_inputs:
      print(f'Transforming input of {col}...')
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    output = predictions.copy()

    column_names = predictions.columns

    for col in column_names:
      if col not in {'forecast_time', 'identifier', 'weights'}:
        output[col] = self._target_scaler.inverse_transform(predictions[[col]])
      elif col == 'weights':
        output[col] = self._real_scalers.inverse_transform(predictions[[col]])
    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 5 + 1,  # 252 + 5,
        'num_encoder_steps': 5,  # 252,
        'num_epochs': 1,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5,
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.3,
        'hidden_layer_size': 80,
        'learning_rate': 0.01,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
        'num_heads': 1,
        'stack_size': 1
    }

    return model_params
