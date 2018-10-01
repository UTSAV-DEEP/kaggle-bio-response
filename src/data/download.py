
import requests
import click
import zipfile
import os

from src.commons import constants


@click.command()
@click.argument('url', required=False,default=constants.RAW_DATA_URL)
@click.argument('filename', type=click.Path(),required=False,default=constants.RAW_ZIP_FILENAME)
def download_file(url, filename):
    full_download_path=constants.BASE_DIR+'/data/raw/'+filename
    print('Downloading from {} to {}'.format(url, full_download_path))
    response = requests.post(url,constants.KAGGLE_INFO)
    with open(full_download_path, 'wb') as ofile:
        ofile.write(response.content)

    print(os.getcwd())
    with zipfile.ZipFile(full_download_path, "r") as zip_ref:
        zip_ref.extractall(constants.BASE_DIR+'/data/raw/')


if __name__ == '__main__':
    download_file()
