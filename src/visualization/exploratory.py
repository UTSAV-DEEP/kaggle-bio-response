import matplotlib
import pandas as pd

matplotlib.use('agg')
import seaborn as sns
import sys

sys.path.append('src')
from src.data import preprocess
from src.commons import constants


def target_correlation_plot(dframe):
    """
    It plots a bar graph between target column and correlation values of all other dimensions with the target column.
    This visualization is chosen because even the processed dataframe for the problem contains 518 feature columns.
    So many plots like pairplot, correlation matrix plot, etc would become very huge and impossible to render.

    :param dframe: dataframe to visualize
    :return: an object of seaborn figure
    """
    sns.set(rc={'figure.figsize': (7, 100)})
    sns.set(font_scale=0.6)
    figure = sns.barplot(dframe.corr()[constants.RESULT_COLUMN_NAME], preprocess.get_headers(dframe)).get_figure()
    sns.reset_defaults()
    return figure


def main():
    print('Plotting activity correlation bar')

    raw_dframe = pd.read_pickle(constants.RAW_DATA_PICKLE)
    plot = target_correlation_plot(raw_dframe)
    plot.savefig(constants.RAW_VISUALIZATION_OUTPUT_FILE)

    processed_dframe = pd.read_pickle(constants.PROCESSED_DATA_FILE)
    plot = target_correlation_plot(processed_dframe)
    plot.savefig(constants.PROCESSED_VISUALIZATION_OUTPUT_FILE)


if __name__ == '__main__':
    main()
