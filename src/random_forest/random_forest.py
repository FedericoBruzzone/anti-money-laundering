import os
from dotenv import load_dotenv
load_dotenv()

from src.utils.kaggle_config import setup_kaggle
from src.utils.kaggle_config import download_dataset
from src.utils.datasets_handler import get_train_and_test
from src.utils.datasets_handler import get_X_and_Y
from src.utils.spark_config import get_spark_session

from src.decision_tree.ID3 import DecisionTreeID3
from pyspark import TaskContext
import pandas as pd

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
    
    COLUMNS_NAME: list = df_train.columns.tolist()
    X_test, y_test = get_X_and_Y(df_test, verbose=VERBOSE)
    X_train, y_train = get_X_and_Y(df_train, verbose=VERBOSE)
    encoder = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    y_test = y_test.replace(encoder)
    y_test[y_test == 2] = 0
    
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
        encoder = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        y_train = y_train.replace(encoder)
        y_train[y_train == 2] = 0

        decision_tree: DecisionTreeID3 = DecisionTreeID3(max_depth=10, 
                                                         num_thresholds_numerical_attr=6)
        decision_tree.fit(X_train, y_train)
        # ctx = TaskContext()
        # decision_tree.create_dot_files(filename="tree" + str(ctx.partitionId()),
        #                                generate_png=True,
        #                                view=VIEW)
        
        yield decision_tree 
    
    def predict_trees(new_line):
        def wrap(tree):
            print("TREE: ", tree)
            print("NEW LINE: ", new_line)
            # prediction = tree.predict(new_line)
            # print("PREDICTION: ", prediction)
            return tree
        return wrap
   
    rdd_tree = rdd.mapPartitions(create_trees)
    print(rdd_tree.collect())
    first_tree = rdd_tree.first() 
    
    print("FIRST TREE: ")
    first_tree.create_dot_files(filename="tree",
                                generate_png=True,
                                view=VIEW)
    new_line = X_test.iloc[0]
    print("NEW LINE: ", new_line)
    # prediction = first_tree.predict(new_line)
    predictions = list(first_tree.predict_test(X_train))
    print("PREDICTIONS: ", predictions)

    # for i in range(len(X_test)):
    #     rdd_predictions = rdd_tree.map(predict_trees(new_line))
    #     print(rdd_predictions.collect())
    #     break
    

    # TEST
    # print("Partition number: ", rdd.getNumPartitions())
    # print(len(rdd.collect()))

    # def process_partition(partition):
    #     # print(type(worker_dataset))
    #     # create tree
    #     partition_elements: list = list()
    #     for element in partition:
    #         # yield element[0]
    #         partition_elements.append(element[0])
    #     yield partition_elements
    
    # rdd1 = rdd.mapPartitions(process_partition)
    
    # print(len(rdd1.collect()))

    # print(rdd1.collect())

    spark.stop()
