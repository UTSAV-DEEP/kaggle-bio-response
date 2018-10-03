import click
import matplotlib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plot

matplotlib.use('agg')
import seaborn as sns

from src.data import preprocess
from src.commons import constants


def exploratory_visualization(dframe):
    sns.set(rc={'figure.figsize': (6, 70)})
    print(dframe.corr()['Activity'])
    print(preprocess.get_headers(dframe))
    return sns.barplot(dframe.corr()['Activity'], preprocess.get_headers(dframe)).get_figure()


@click.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False), required=False,
                default=constants.PROCESSED_DATA_FILE)
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False), required=False,
                default=constants.VISUALIZATION_OUTPUT_FILE)
def main(input_file, output_file):
    print('Plotting activity correlation bar')

    dframe = preprocess.read_processed_data(input_file)
    plot = exploratory_visualization(dframe)
    plot.savefig(output_file)


if __name__ == '__main__':
    main()