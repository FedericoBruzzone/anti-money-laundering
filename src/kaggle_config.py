import os
from dotenv import load_dotenv
load_dotenv()

def setup_kaggle():
    kaggle_config_dir = os.path.join(os.getcwd(), ".kaggle")

    if (os.path.exists(kaggle_config_dir)):
        os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config_dir
    else:
        KAGGLE_USER = os.getenv("KAGGLE_USER")
        KAGGLE_KEY = os.getenv('KAGGLE_KEY')
        
        os.environ['KAGGLE_USERNAME'] = KAGGLE_USER
        os.environ['KAGGLE_KEY'] = KAGGLE_KEY

def download_dataset(dataset_name="", path='datasets/', unzip=True):
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    kaggle_dataset = os.path.join(os.getcwd(), "datasets")

    if (not os.path.exists(kaggle_dataset)):
        if dataset_name == "":
            dataset_name = 'ealtman2019/ibm-transactions-for-anti-money-laundering-aml' 

        kaggle_api = KaggleApi()
        kaggle_api.authenticate()
        kaggle_api.dataset_download_files(dataset_name, path=path, unzip=unzip)

