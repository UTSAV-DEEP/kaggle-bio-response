

import click
import numpy as np
import os

from sklearn import metrics
import matplotlib.pyplot as plt

from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel


from src.data import preprocess
from src.commons import constants


def plot_roc_curve(np_actual, np_predicted):
    fpr, tpr, thresholds = metrics.roc_curve(np_actual, np_predicted)
    plt.plot(fpr, tpr, 'r-', label='trained model')
    plt.plot([0, 1], [0, 1], 'k-', label='random')
    plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


def print_evaluation_metrics(actual_labels, predicted_labels):
    np_actual=np.array(actual_labels)
    np_predicted=np.array(predicted_labels)

    print(metrics.confusion_matrix(np_actual, np_predicted))
    print(metrics.accuracy_score(np_actual, np_predicted))
    print(metrics.recall_score(np_actual, np_predicted))
    print(metrics.precision_score(np_actual, np_predicted))
    print(metrics.f1_score(np_actual, np_predicted))
    print(metrics.roc_auc_score(np_actual, np_predicted))
    plot_roc_curve(np_actual, np_predicted)


def get_model(model):
    model_file = os.path.join(constants.PROJ_ROOT, 'models', model.name + '.model')
    model.load(model_file)
    return model


def show_models_performances(models,test_x,test_y):
    for model in models:
        print('Performance of '+model.name)
        predicted_y = model.predict(test_x)
        print_evaluation_metrics(test_y, predicted_y)


@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False), required=False,
                default=constants.PROCESSED_DATA_FILE)
def main(input_file):
    dframe = preprocess.read_processed_data(input_file)
    train, test = preprocess.get_train_test_split(dframe)

    test_x = preprocess.get_featues(test)
    test_y = preprocess.get_label(test)

    show_models_performances([get_model(RandomForestModel()),
                              get_model(LogisticRegressionModel())],
                             test_x,test_y)


if __name__ == '__main__':
    main()
