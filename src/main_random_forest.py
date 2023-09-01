import os
from dotenv import load_dotenv

from src.utils.dataset_sampling_methods import (oversampling,
                                                undersampling,
                                                bootstrap_sampling)
load_dotenv()
from collections import Counter

from src.utils.kaggle_config import setup_kaggle
from src.utils.kaggle_config import download_dataset
from src.utils.datasets_handler import get_train_and_test
from src.utils.datasets_handler import get_X_and_Y
from src.utils.spark_config import get_spark_session
from src.utils.datasets_handler import label_encoder
from src.decision_tree.ID3 import DecisionTreeID3
from src.utils.performance_measures import calculate_performances

from pyspark import TaskContext
import pandas as pd

def create_trees(partition_elements, verbose=False):
    list_series = []
    for element in partition_elements:
        series_tmp = pd.Series(element.asDict())
        list_series.append(series_tmp)

    part_df = pd.DataFrame(list_series, columns=COLUMNS_NAME)
    X_train, y_train = get_X_and_Y(part_df, verbose=VERBOSE)
    X_train, _ = label_encoder(X_train, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])

    decision_tree: DecisionTreeID3 = DecisionTreeID3(max_depth=8,
                                                     num_thresholds_numerical_attr=2,
                                                     VERBOSE=False)
    decision_tree.fit(X_train, y_train)

    if verbose:
        ctx = TaskContext()
        decision_tree.create_dot_files(filename="tree" + str(ctx.partitionId()),
                                       generate_png=True,
                                       view="default-viewer")
    yield decision_tree

def predict_trees(new_line):
    def wrap(tree):
        prediction = tree.predict(new_line)
        return prediction
    return wrap

def predict_trees_all(X_test):
    def wrap(tree):
        predictions = tree.predict_test_no_gen(X_test)
        return predictions
    return wrap

if __name__ == "__main__":
    VERBOSE = int(os.getenv('VERBOSE'))
    VIEW = os.getenv('VIEW')
    name = "AntiMoneyLaundering"

    spark = get_spark_session(name, VERBOSE)

    setup_kaggle()
    print("---------------------- Downloading dataset ----------------------")
    download_dataset("uciml/iris")
    download_dataset("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
    print("---------------------- End downloading dataset ----------------------")

    hi_small_trans = "HI-Small_Trans.csv"
    iris = "Iris.csv"

    df_train, df_test = get_train_and_test(hi_small_trans, verbose=VERBOSE)

    df_train = bootstrap_sampling(df_train)
    # df_train = oversampling(df_train, VERBOSE=False)

    COLUMNS_NAME: list = df_train.columns.tolist()
    X_train, y_train = get_X_and_Y(df_train, verbose=VERBOSE)
    X_test, y_test = get_X_and_Y(df_test, verbose=VERBOSE)

    # X_train, _ = label_encoder(X_train, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
    X_test, _ = label_encoder(X_test, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])

    df = spark.createDataFrame(df_train)

    print("Printing spark dataframe...")
    df.show()

    rdd = df.rdd

    def map_to_column_value_pairs(row):
        return [(i, row[i]) for i in range(len(row))]

    def count_values(a, b):
        return a + b

    predictions = rdd.mapPartitions(create_trees, False) \
                     .map(predict_trees_all(X_test)) \
                     .flatMap(map_to_column_value_pairs) \
                     .map(lambda x: (x, 1)) \
                     .reduceByKey(count_values) \
                     .map(lambda x: (x[0][0], [(x[0][1], x[1])])) \
                     .reduceByKey(count_values) \
                     .map(lambda x: (x[0], max(x[1], key=lambda item: item[1]))) \
                     .map(lambda x: x[1][0]) \
                     .collect()

    print(predictions)

    calculate_performances(predictions, y_test, verbose=True)

    # rdd_tree = rdd.mapPartitions(create_trees, False).cache().persist()
    # rdd_tree.collect()
    #
    # print("End creating trees")
    #
    # import time
    # list_predictions = []
    # for i in range(len(X_test)):
    #     s1 = time.time()
    #     tree_predictions = rdd_tree.map(predict_trees(X_test.iloc[i])).collect()
    #     e1 = time.time()
    #     print("Time to predict: ", e1 - s1)
    #
    #     e3 = time.time()
    #     list_predictions.append(Counter(tree_predictions).most_common(1)[0][0])
    #     e4 = time.time()
    #     print("Time to get most common: ", e4 - e3)
    #
    #     print(i)
    #
    # print("End predicting trees")
    #
    # calculate_performances(list_predictions, y_test, verbose=True)

    spark.stop()

