import os
from dotenv import load_dotenv

load_dotenv()

kaggle_config_dir = os.path.join(os.getcwd(), ".kaggle")

os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config_dir

KAGGLE_USER = os.getenv("KAGGLE_USER")
KAGGLE_KEY = os.getenv('KAGGLE_KEY')

os.environ['KAGGLE_USERNAME'] = KAGGLE_USER
os.environ['KAGGLE_KEY'] = KAGGLE_KEY

from kaggle.api.kaggle_api_extended import KaggleApi

kaggle_api = KaggleApi()
kaggle_api.authenticate()
kaggle_api.dataset_download_files('ealtman2019/ibm-transactions-for-anti-money-laundering-aml', path='datasets/', unzip=True)
