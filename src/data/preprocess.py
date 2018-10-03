import click
import pandas as pd
from src.commons import constants
from sklearn.decomposition import PCA
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
    return dframe


def preprocess_data(dframe):
    dframe = dframe.copy()
    dframe = dframe.loc[:, dframe.std() > .01]
    dframe = dframe.loc[:, abs(dframe.corr()[constants.RESULT_COLUMN_NAME]) > .05]
    label = get_label(dframe)
    features = get_featues(dframe)
    # features = preprocessing.StandardScaler().fit_transform(features)
    # pca = PCA(0.95)
    # pca.fit(features)
    # features=pca.transform(features)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(features)
    features = pd.DataFrame(x_scaled)
    features.insert(0,constants.RESULT_COLUMN_NAME,label)
    return features


def read_processed_data(fname=constants.PROCESSED_DATA_FILE):
    dframe = pd.read_pickle(fname)
    return dframe


def get_train_validation_test_split(dframe):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    result = next(skf.split(dframe, get_label(dframe)), None)

    train_validation = dframe.iloc[result[0]]
    test = dframe.iloc[result[1]]

    result = next(skf.split(train_validation, get_label(train_validation)), None)

    train = train_validation.iloc[result[0]]
    validation = train_validation.iloc[result[1]]
    return train,validation,test


def get_train_test_split(dframe):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
    result = next(skf.split(dframe, get_label(dframe)), None)

    train = dframe.iloc[result[0]]
    test = dframe.iloc[result[1]]

    return train,test


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False), required=False,
                default=constants.RAW_DATA_FILE)
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False), required=False,
                default=constants.PROCESSED_DATA_FILE)
@click.option('--excel', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file, excel):
    print('Preprocessing data')

    dframe = read_raw_data(input_file)


    dframe = preprocess_data(dframe)

    dframe.to_pickle(output_file)
    if excel is not None:
        dframe.to_excel(excel)

    train,validation,test = get_train_validation_test_split(dframe)

    print(train.shape)
    print(validation.shape)
    print(test.shape)



if __name__ == '__main__':
    main()
