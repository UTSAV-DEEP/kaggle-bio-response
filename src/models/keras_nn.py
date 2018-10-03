from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml

from src.data.preprocess import get_featues, get_label


class KerasNN(object):
    def __init__(self):
        self.clf = Sequential()
        self.clf.add(Dense(20, input_dim=518, activation='relu'))
        self.clf.add(Dense(30, activation='relu'))
        self.clf.add(Dense(1, activation='sigmoid'))
        self.clf.compile(loss='binary_crossentropy', optimizer='adam')
        self.name = 'kerasNN'

    def train(self, dframe):
        X = get_featues(dframe)
        y = get_label(dframe)
        self.clf.fit(X, y, epochs=500, verbose=0)

    def predict(self, X):
        y_pred = self.clf.predict(X)
        y_pred=(y_pred>0.5)*1
        return y_pred

    def save(self, fname):
        model_yaml = self.clf.to_yaml()
        yaml_file_name = fname + '.yaml'
        with open(yaml_file_name, "w") as yaml_file:
            yaml_file.write(model_yaml)
            # serialize weights to HDF5
            weights_file_name = fname + '.h5'
            self.clf.save_weights(weights_file_name)

    def load(self, fname):
        yaml_file_name = fname + '.yaml'
        with open(yaml_file_name, 'r') as yaml_file:
            loaded_model_yaml = yaml_file.read()
            self.clf = model_from_yaml(loaded_model_yaml)
            # load weights into new model
            weights_file_name = fname + '.h5'
            self.clf.load_weights(weights_file_name)
