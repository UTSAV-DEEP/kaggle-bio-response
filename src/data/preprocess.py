import sys

import click
import pandas as pd
import numpy as np

sys.path.append('src')
from src.commons import constants
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


def get_headers(dframe):
    return list(dframe)


def get_featues(dframe):
    return dframe[get_headers(dframe)[1:]]


def get_label(dframe):
    return dframe[constants.RESULT_COLUMN_NAME]


def read_raw_data(fname=constants.RAW_DATA_FILE):
    dframe = pd.read_csv(fname)
    dframe.to_pickle(constants.RAW_DATA_PICKLE)
    return dframe


def preprocess_data(dframe,out_file=constants.PROCESSED_DATA_FILE):
    """
    It returns a processed dataframe without modifying the original one. It also saves the processed data pickle

    The training dataset for this problem do not contain any NaN, missing or outliers.
    All the data were in range between 0 and 1.
    However, there are many columns that contain more or less constant values.
    Also many columns have close to 0 correlation with the target column. These columns are removed in this method.
    After that the data is then transformed with StandardScaler to make column's mean value 0 and std. dev. as 1.

    This method successfully reduces the dataframe's dimensions (excluding target column) from 1776 to 518 without
    compromising on evaluation metrics

    :param dframe: source dataframe
    :param out_file: file path to save processed data pickle
    :return: processed copy of the given dataframe
    """
    dframe = dframe.copy()
    # removing dimensions with std. dev. < 0.01 as these are almost constant columns
    dframe = dframe.loc[:, dframe.std() > .01]

    # removing dimensions that have low absolute correlation with target column as these are possibly noise
    dframe = dframe.loc[:, abs(dframe.corr()[constants.RESULT_COLUMN_NAME]) > .05]

    # sorting the dataframe columns in descending order based on their absolute correlation with target variable
    dframe = dframe.iloc[:, np.argsort(-abs(dframe.corr()[constants.RESULT_COLUMN_NAME]))]
    headers = get_headers(dframe)
    # print(headers)
    # standardising the dataframe so that all columns have mean 0 and std = 1
    dframe[headers[1:]] = preprocessing.StandardScaler().fit_transform(dframe[headers[1:]])
    # print(dframe.describe())

    dframe.to_pickle(out_file)
    return dframe


def read_processed_data(fname=constants.PROCESSED_DATA_FILE):
    dframe = pd.read_pickle(fname)
    return dframe


def get_train_validation_test_split(dframe):
    """
    This method splits the given dataframe object into train,validation and test dataframes in stratified manner.

    :param dframe: source dataframe
    :return: a tuple containing train,validation and test dataframes
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    result = next(skf.split(dframe, get_label(dframe)), None)

    train_validation = dframe.iloc[result[0]]
    test = dframe.iloc[result[1]]

    result = next(skf.split(train_validation, get_label(train_validation)), None)

    train = train_validation.iloc[result[0]]
    validation = train_validation.iloc[result[1]]
    return train, validation, test


def get_train_test_split(dframe):
    """
    This method splits the given dataframe object into train and test dataframes in stratified manner.

    :param dframe: source dataframe
    :return: a pair containing train and test dataframes
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    result = next(skf.split(dframe, get_label(dframe)), None)

    train = dframe.iloc[result[0]]
    test = dframe.iloc[result[1]]

    return train, test



@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False), required=False,
                default=constants.RAW_DATA_FILE)
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False), required=False,
                default=constants.PROCESSED_DATA_FILE)
@click.option('--excel', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file, excel):
    print('Preprocessing data')

    dframe = read_raw_data(input_file)
    print(dframe.head())
    dframe = preprocess_data(dframe,out_file=constants.PROCESSED_DATA_FILE)

    if excel is not None:
        dframe.to_excel(excel)


if __name__ == '__main__':
    main()
