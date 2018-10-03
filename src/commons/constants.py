import os

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
PROJ_ROOT = os.path.join(PROJ_ROOT, '..')
RAW_DATA_FILE = os.path.join(PROJ_ROOT,'data', 'raw', 'train.csv')

RAW_DATA_URL='https://www.kaggle.com/c/bioresponse/download/train.csv'

PROCESSED_DATA_FILE=os.path.join(PROJ_ROOT, 'data', 'processed', 'processed.pickle')
RESULT_COLUMN_NAME='Activity'
VISUALIZATION_OUTPUT_FILE=os.path.join(PROJ_ROOT, 'reports', 'figures', 'visual.png')