import os
import sys

import click
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

sys.path.append('src')
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.svm import SVM
from src.models.keras_nn import KerasNN

from src.data import preprocess
from src.commons import constants


def print_evaluation_metrics(actual_labels, predicted_labels, predicted_y_prob):
    np_actual = np.array(actual_labels)
    np_predicted = np.array(predicted_labels)
    np_predicted_proba = np.array(predicted_y_prob)

    print("\nconfusion_matrix:\n" + str(metrics.confusion_matrix(np_actual, np_predicted)))
    print("\nclassification_report:\n" + metrics.classification_report(np_actual, np_predicted))
    print("\naccuracy_score:\t%.3f" % metrics.accuracy_score(np_actual, np_predicted))
    print("\nroc_auc_score:\t%.3f" % metrics.roc_auc_score(np_actual, np_predicted))
    print("\nlog_loss:\t%.3f" % metrics.log_loss(np_actual, np_predicted_proba))


def get_model(model):
    model_file = os.path.join(constants.PROJ_ROOT, 'models', model.name + '.model')
    model.load(model_file)
    return model


def get_model_keras(model=KerasNN()):
    model_file_pref = os.path.join(constants.PROJ_ROOT, 'models', model.name)
    model.load(model_file_pref)
    return model


def show_models_performances(models, test_x, test_y, roc_file=constants.ROC_CURVES_PATH):
    plt.grid(color='y', linestyle='-', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')
    plt.plot([0, 1], [0, 1], 'k-', label='random')
    plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
    colors = ['b-', 'r-', 'c-', 'm-', 'y-']
    idx = 0
    for model in models:
        print('_____________________________________________________________________________________')
        print('\nPerformance of ' + model.name)
        print('-------------------------------------------------------------------------------------')
        predicted_y = model.predict(test_x)
        predicted_y_prob = model.predict_proba(test_x)
        print_evaluation_metrics(test_y, predicted_y, predicted_y_prob)
        np_actual = np.array(test_y)
        np_predicted = np.array(predicted_y)
        fpr, tpr, thresholds = metrics.roc_curve(np_actual, np_predicted)
        plt.plot(fpr, tpr, colors[idx], label=model.name)
        idx += 1
        print('_____________________________________________________________________________________')

    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(roc_file)
    plt.clf()


def show_benchmark_model_performance():
    print('Calculating benchmark model performance:')
    dframe = preprocess.read_processed_data(constants.RAW_DATA_PICKLE)
    train, test = preprocess.get_train_test_split(dframe)

    test_x = preprocess.get_featues(test)
    test_y = preprocess.get_label(test)
    model = SVM()
    model_file = os.path.join(constants.PROJ_ROOT, 'models', 'benchmark', model.name + '.model')
    model.load(model_file)
    show_models_performances([model], test_x, test_y, roc_file=constants.BENCHMARK_ROC_PATH)


def show_improved_model_performances(dframe):
    print('\nCalculating improved models performances:')
    train, test = preprocess.get_train_test_split(dframe)

    test_x = preprocess.get_featues(test)
    test_y = preprocess.get_label(test)

    show_models_performances([get_model(RandomForestModel()),
                              get_model(LogisticRegressionModel()),
                              get_model(SVM()),
                              get_model_keras()],
                             test_x, test_y)

@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False), required=False,
                default=constants.PROCESSED_DATA_FILE)
def main(input_file):
    show_benchmark_model_performance()
    dframe = preprocess.read_processed_data(input_file)
    show_improved_model_performances(dframe)


if __name__ == '__main__':
    main()
