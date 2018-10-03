from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml
import sys

sys.path.append('src')
from src.data.preprocess import get_featues, get_label


class KerasNN(object):
    def __init__(self):
        self.clf = Sequential()
        self.clf.add(Dense(20, input_dim=518, activation='relu'))
        self.clf.add(Dense(20, activation='relu'))
        self.clf.add(Dense(1, activation='sigmoid'))
        self.clf.compile(loss='binary_crossentropy', optimizer='adam')
        self.name = 'KerasNN'

    def train(self, train, validation):
        X = get_featues(train)
        y = get_label(train)
        xv = get_featues(validation)
        yv = get_label(validation)
        self.clf.fit(X, y, epochs=1000, verbose=0, validation_data=(xv,yv))

    def predict(self, X):
        y_pred = self.clf.predict(X)
        y_pred=(y_pred>0.5)*1
        return y_pred

    def save(self, fname):
        model_yaml = self.clf.to_yaml()
        yaml_file_name = fname + '.yaml'
        with open(yaml_file_name, "w+") as yaml_file:
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
