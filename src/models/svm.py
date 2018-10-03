import pickle
from sklearn.svm import SVC
import sys

sys.path.append('src')
from src.data.preprocess import get_featues, get_label


class SVM(object):
    def __init__(self):
        self.clf = SVC(probability=True)
        self.name = 'SVM'

    def get_params(self):
        return self.clf.get_params()

    def train(self, dframe):
        X = get_featues(dframe)
        y = get_label(dframe)
        self.clf.fit(X, y)

    def predict(self, X):
        y_pred = self.clf.predict(X)

        return y_pred

    def save(self, fname):
        with open(fname, 'wb') as ofile:
            pickle.dump(self.clf, ofile, pickle.HIGHEST_PROTOCOL)

    def load(self, fname):
        with open(fname, 'rb') as ifile:
            self.clf = pickle.load(ifile)
