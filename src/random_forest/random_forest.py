import os
from dotenv import load_dotenv
load_dotenv()

from src.utils.kaggle_config import setup_kaggle
from src.utils.kaggle_config import download_dataset
from src.utils.datasets_handler import get_train_and_test
from src.utils.datasets_handler import get_X_and_Y

from src.utils.spark_config import get_spark_session

if __name__ == "__main__":
    VERBOSE = int(os.getenv('VERBOSE'))
    VIEW = os.getenv('VIEW')
    name = "AntiMoneyLaundering"

    spark = get_spark_session(name, VERBOSE)
    
    setup_kaggle()
    print("Downloading dataset...")
    download_dataset()
    print("Done!")

    hi_small_trans = "HI-Small_Trans.csv"
    iris = "Iris.csv"

    df_train, df_test = get_train_and_test(iris, verbose=VERBOSE)
    X_train, Y_train = get_X_and_Y(df_train, verbose=VERBOSE)
    X_test, Y_test = get_X_and_Y(df_test, verbose=VERBOSE)

    df = spark.createDataFrame(X_train)

    print("Printing spark dataframe...")
    df.show()

    rdd = df.rdd
   
    print(len(rdd.collect()))

    def process_partition(worker_dataset):
        # print(type(worker_dataset))
        # create tree
        partition_element: list = list()
        for element in worker_dataset:
            # yield element[0]
            partition_element.append(element[0])
        yield partition_element
    
    rdd1 = rdd.mapPartitions(process_partition)
    
    print(len(rdd1.collect()))

    print(rdd1.collect())

