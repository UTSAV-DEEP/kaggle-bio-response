import click
import os

from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.data import preprocess
from src.commons import constants


def train_save_model(model,dframe):
    model.train(dframe)
    model_file_name=os.path.join(constants.PROJ_ROOT, 'models', model.name + '.model')
    model.save(model_file_name)

@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False), required=False,
                default=constants.PROCESSED_DATA_FILE)
def main(input_file):
    print('Training model')

    dframe = preprocess.read_processed_data(input_file)

    train,test=preprocess.get_train_test_split(dframe)
    train_save_model(RandomForestModel(), train)
    train_save_model(LogisticRegressionModel(), train)


if __name__ == '__main__':
    main()