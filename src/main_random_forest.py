import os
from dotenv import load_dotenv
load_dotenv()

from src.kaggle_config import setup_kaggle
from src.kaggle_config import download_dataset
from src.utils.datasets_handler import get_train_and_test
from src.utils.datasets_handler import get_X_and_Y
from src.utils.datasets_handler import print_dataset

from src.decision_tree.decision_tree import print_dataset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    VERBOSE = int(os.getenv('VERBOSE'))

    setup_kaggle()
    print("Downloading dataset...")
    download_dataset()
    print("Done!")

    df_train, df_test = get_train_and_test(verbose=VERBOSE)
    X_train, Y_train = get_X_and_Y(df_train, verbose=VERBOSE)
    X_test, Y_test = get_X_and_Y(df_test, verbose=VERBOSE)
    
    # print_dataset(X_train, Y_train)
    

    # print(type(X_train.iloc[0].values)) # <class 'numpy.ndarray'>
    # print(X_train.iloc[0]["Timestamp"])
