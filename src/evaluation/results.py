import os
import sys

import click
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import pandas as pd

sys.path.append('src')
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.svm import SVM
from src.models.keras_nn import KerasNN

from src.data import preprocess
from src.commons import constants


def get_model(model):
    model_file = os.path.join(constants.PROJ_ROOT, 'models', model.name + '.model')
    model.load(model_file)
    return model


def get_model_keras(model=KerasNN()):
    model_file_pref = os.path.join(constants.PROJ_ROOT, 'models', model.name)
    model.load(model_file_pref)
    return model


def generate_results(models,test,res_dir=constants.TEST_RESULT_CSV_DIR):
    for model in models:
        print('_____________________________________________________________________________________')
        print('\nPerformance of ' + model.name)
        print('-------------------------------------------------------------------------------------')
        predicted_y = model.predict(test)
        predicted_y_prob = model.predict_proba(test)[:,-1]
        n=predicted_y_prob.shape[0]
        res=pd.DataFrame(data=predicted_y_prob,columns=['PredictedProbability'])
        res.insert(loc=0,column='MoleculeId',value = res.index+1)
        res.to_csv(os.path.join(res_dir,model.name+'.csv'),index=False)


@click.command()
@click.argument('test_file', type=click.Path(exists=True, dir_okay=False), required=False,
                default=constants.PROCESSED_TEST_FILE)
def main(test_file):
    test = preprocess.read_processed_data(test_file)
    generate_results([get_model(RandomForestModel()),
                      get_model(LogisticRegressionModel()),
                      get_model(SVM()),
                      get_model_keras()],test)



if __name__ == '__main__':
    main()