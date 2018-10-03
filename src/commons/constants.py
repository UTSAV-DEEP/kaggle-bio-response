import os

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
PROJ_ROOT = os.path.join(PROJ_ROOT, '..')
RAW_DATA_FILE = os.path.join(PROJ_ROOT,'data', 'raw', 'train.csv')

RAW_DATA_URL='https://www.kaggle.com/c/bioresponse/download/train.csv'

RAW_DATA_PICKLE=os.path.join(PROJ_ROOT, 'data', 'raw', 'raw.pickle')
PROCESSED_DATA_FILE=os.path.join(PROJ_ROOT, 'data', 'processed', 'processed.pickle')
RESULT_COLUMN_NAME='Activity'
VISUALIZATION_OUTPUT_FILE=os.path.join(PROJ_ROOT, 'reports', 'figures', 'target_corr_plot.png')
ROC_CURVES_PATH=os.path.join(PROJ_ROOT, 'reports', 'figures','evaluation', 'roc_curves.png')
BENCHMARK_ROC_PATH=os.path.join(PROJ_ROOT, 'reports', 'figures','evaluation', 'benchmark_roc_curve.png')