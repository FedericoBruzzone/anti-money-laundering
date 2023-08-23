import os
from dotenv import load_dotenv
load_dotenv()

import time
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.utils.kaggle_config            import setup_kaggle
from src.utils.kaggle_config            import download_dataset

from src.utils.datasets_handler         import get_train_and_test
from src.utils.datasets_handler         import get_X_and_Y
from src.utils.datasets_handler         import print_dataset
from src.utils.datasets_handler         import label_encoder
from src.utils.performance_measures     import calculate_performances
from src.utils.plot_measures            import (plot_correlation_matrix,  
                                                plot_numerical_histograms, 
                                                plot_roc_curve,
                                                plot_confusion_matrix)
from src.utils.dataset_sampling_methods import (oversampling,
                                                undersampling,
                                                bootstrap_sampling)

from src.decision_tree.decision_tree    import CustomDecisionTree
from src.decision_tree.ID3              import DecisionTreeID3
from src.decision_tree.entropy_type     import EntropyType
from src.decision_tree.criterion_type   import CriterionType

from IPython.display import Image, display

VERBOSE = int(os.getenv('VERBOSE'))
VIEW = os.getenv('VIEW')

setup_kaggle()
print("---------------------- Downloading dataset ----------------------") 
download_dataset("iammustafatz/diabetes-prediction-dataset")
download_dataset("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
print("---------------------- End downloading dataset ----------------------")

hi_small_trans = "HI-Small_Trans.csv"
diabetes = "diabetes_prediction_dataset.csv"

original_df_train, original_df_test = get_train_and_test(hi_small_trans, verbose=VERBOSE)
original_df_train, _ = label_encoder(original_df_train, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
original_df_test, _ = label_encoder(original_df_test, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])

hp_n_thresholds_values = [2, 4, 6]

def id3_experiment(df_train, df_test, type):
    X_train, y_train = get_X_and_Y(df_train, verbose=VERBOSE)
    X_test, y_test = get_X_and_Y(df_test, verbose=VERBOSE)
    
    for hp_n_thresholds in hp_n_thresholds_values:
        print(f"\n\033[32mNumber of thresholds: {hp_n_thresholds} \033[0m")
        start_time = time.time()
        decision_tree: DecisionTreeID3 = DecisionTreeID3(max_depth=10, num_thresholds_numerical_attr=hp_n_thresholds)
        decision_tree.fit(X_train, y_train)
        end_time = time.time()
        decision_tree.create_dot_files(filename=f"tree-id3-{type}-{hp_n_thresholds}", generate_png=True, view=VIEW)
        print("PERFORMANCES: ")
        predictions = list(decision_tree.predict_test(X_test))
        
        calculate_performances(predictions, y_test, "id3", verbose=True)

        print(f"Fit time: {end_time - start_time} seconds")

print("\nWithout preprocessing")
df_train, df_test = original_df_train, original_df_test
id3_experiment(df_train, df_test, "wo_preprocessing")