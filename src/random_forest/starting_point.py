import os
from dotenv import load_dotenv
load_dotenv()

from src.kaggle_config import setup_kaggle
from src.kaggle_config import download_dataset
from src.utils.datasets_handler import get_train_and_test
from src.utils.datasets_handler import get_X_and_Y

from pyspark.sql import SparkSession
import pyspark

import warnings

# Suppress PySpark warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    VERBOSE = int(os.getenv('VERBOSE'))

    setup_kaggle()
    print("Downloading dataset...")
    download_dataset()
    print("Done!")

    hi_small_trans = "HI-Small_Trans.csv"
    iris = "Iris.csv"
    
    df_train, df_test = get_train_and_test(iris, verbose=VERBOSE)
    X_train, Y_train = get_X_and_Y(df_train, verbose=VERBOSE)
    X_test, Y_test = get_X_and_Y(df_test, verbose=VERBOSE)
    
    # print(type(X_train.iloc[0].values)) # <class 'numpy.ndarray'>
    # print(X_train.iloc[0]["Timestamp"])
    
    # ====== Spark ======
    sc = pyspark.SparkContext('local[3]').getOrCreate() # *
    spark = SparkSession(sparkContext=sc) #.builder.appName("pandas to spark").getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    # spark.conf.set("spark.executor.workerThreads", "2")
    df = spark.createDataFrame(X_train.head())

    print("Printing spark dataframe...")
    df.show()

    rdd = df.rdd

    # Get the number of partitions
    num_partitions = rdd.getNumPartitions()

    print(num_partitions)

    # Collect and print the data distribution across partitions
    data_distribution = rdd.glom().collect()
    for i, partition_data in enumerate(data_distribution):
        print(f"Partition {i}: {len(partition_data)} rows")
        # print(partition_data)

    # Print the total number of rows in the DataFrame
    total_rows = df.count()
    print(f"Total rows: {total_rows}")

    def process_partition(worder_dataset):
        for element in worder_dataset:
            yield element[0]
        # return [element[0] for element in worder_dataset]

    rdd1 = rdd.mapPartitions(process_partition)
    
    print(rdd1.collect())
