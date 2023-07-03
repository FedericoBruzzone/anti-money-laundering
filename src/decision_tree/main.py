from src.kaggle_config import setup_kaggle
from src.kaggle_config import download_dataset

if __name__ == "__main__":
    setup_kaggle()
    print("Downloading dataset...")
    download_dataset()
    print("Done!")

