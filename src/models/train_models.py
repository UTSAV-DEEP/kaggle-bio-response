import os
import sys

import click

sys.path.append('src')
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.svm import SVM
from src.models.keras_nn import KerasNN
from src.data import preprocess
from src.commons import constants


def train_save_model(model, dframe):
    model.train(dframe)
    model_file_name = os.path.join(constants.PROJ_ROOT, 'models', model.name + '.model')
    model.save(model_file_name)


def train_save_keras_model(train,validation):
    model = KerasNN()
    model.train(train,validation)
    model_file_pref = os.path.join(constants.PROJ_ROOT, 'models', model.name)
    model.save(model_file_pref)


def train_save_benchmark_model():
    dframe = preprocess.read_processed_data(constants.RAW_DATA_PICKLE)
    train, test = preprocess.get_train_test_split(dframe)
    model = SVM()
    model.train(train)
    model_file_name = os.path.join(constants.PROJ_ROOT, 'models', 'benchmark', model.name + '.model')
    model.save(model_file_name)


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False), required=False,
                default=constants.PROCESSED_DATA_FILE)
def main(input_file):
    print('Training model')
    train_save_benchmark_model()

    dframe = preprocess.read_processed_data(input_file)

    train, test = preprocess.get_train_test_split(dframe)
    train_save_model(RandomForestModel(), train)
    train_save_model(LogisticRegressionModel(), train)
    train_save_model(SVM(), train)

    train,validation,test = preprocess.get_train_validation_test_split(dframe)
    train_save_keras_model(train,validation)


if __name__ == '__main__':
    main()
