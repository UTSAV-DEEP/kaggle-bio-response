import pickle
import sys
from sklearn.ensemble import RandomForestClassifier

sys.path.append('src')

from src.data.preprocess import get_featues, get_label


class RandomForestModel(object):
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=50, max_depth=700)
        self.name = 'RandomForest'

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
