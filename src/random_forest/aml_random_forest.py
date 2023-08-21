import os
from dotenv import load_dotenv

from src.utils.dataset_sampling_methods import bootstrap_sampling
load_dotenv()
from collections import Counter

from src.utils.kaggle_config import setup_kaggle
from src.utils.kaggle_config import download_dataset
from src.utils.datasets_handler import get_train_and_test
from src.utils.datasets_handler import get_X_and_Y
from src.utils.spark_config import get_spark_session
from src.utils.datasets_handler import label_encoder
from src.decision_tree.ID3 import DecisionTreeID3

from pyspark import TaskContext
import pandas as pd

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
    
    COLUMNS_NAME: list = df_train.columns.tolist()
    X_train, y_train = get_X_and_Y(df_train, verbose=VERBOSE)
    X_test, y_test = get_X_and_Y(df_test, verbose=VERBOSE)


    # X_train, _ = label_encoder(X_train, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
    X_test, _ = label_encoder(X_test, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
    

    df = spark.createDataFrame(df_train)

    print("Printing spark dataframe...")
    df.show()

    rdd = df.rdd

    def create_trees(partition_elements):
        list_series = []
        for element in partition_elements:
            series_tmp = pd.Series(element.asDict())
            list_series.append(series_tmp)

        part_df = pd.DataFrame(list_series, columns=COLUMNS_NAME)
        X_train, y_train = get_X_and_Y(part_df, verbose=VERBOSE)
        X_train, _ = label_encoder(X_train, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])

        decision_tree: DecisionTreeID3 = DecisionTreeID3(max_depth=10, 
                                                         num_thresholds_numerical_attr=6,
                                                         VERBOSE=False)
        decision_tree.fit(X_train, y_train)
        # ctx = TaskContext()
        # decision_tree.create_dot_files(filename="tree" + str(ctx.partitionId()),
        #                                generate_png=True,
        #                                view=VIEW)
        yield decision_tree 
    
    def predict_trees(new_line):
        def wrap(tree):
            prediction = tree.predict(new_line)
            return prediction
        return wrap
   
    rdd_tree = rdd.mapPartitions(create_trees).cache().persist()
    rdd_tree.collect()
    
    for i in range(len(X_test)):
        rdd_predictions = rdd_tree.map(predict_trees(X_test.iloc[i])).cache().persist()
        list_predictions = rdd_predictions.collect()
        print("List predictions: ", list_predictions)
        prediction = Counter(list_predictions).most_common(1)[0][0]
        print("Prediction: ", prediction)

    spark.stop()

