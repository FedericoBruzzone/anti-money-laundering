import os
from dotenv import load_dotenv
load_dotenv()

import time
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.kaggle_config               import setup_kaggle
from src.kaggle_config               import download_dataset

from src.utils.datasets_handler      import get_train_and_test
from src.utils.datasets_handler      import get_X_and_Y
from src.utils.datasets_handler      import print_dataset
from src.utils.datasets_handler      import label_encoder
from src.utils.performance_measures  import calculate_performances
from src.utils.plot_measures         import (plot_correlation_matrix,  
                                             plot_numerical_histograms, 
                                             plot_roc_curve,
                                             plot_confusion_matrix)

from src.decision_tree.decision_tree  import CustomDecisionTree
from src.decision_tree.ID3            import DecisionTreeID3
from src.decision_tree.entropy_type   import EntropyType
from src.decision_tree.criterion_type import CriterionType

if __name__ == "__main__":
    VERBOSE = int(os.getenv('VERBOSE'))
    VIEW = os.getenv('VIEW')

    setup_kaggle()
    print("Downloading dataset")
    download_dataset("uciml/iris")
    print("End downloading dataset")
    
    hi_small_trans = "HI-Small_Trans.csv"
    iris = "Iris.csv"
    df_train, df_test = get_train_and_test(hi_small_trans, verbose=VERBOSE)

    # Undersampling --------
    # print("df len: ", len(df_train))
    # is_laundering = df_train[df_train['Is Laundering']==1]
    # is_not_laundering = df_train[df_train['Is Laundering']==0]
    # is_not_laundering = is_not_laundering.sample(n=len(is_laundering), random_state=101)
    # df_train = pd.concat([is_laundering, is_not_laundering],axis=0)
    # print("Undersampled df len: ", len(df_train))
    # ----------------------
    


    # OVERSAMPLING
    # ----------------------------------------------------------------------------
    print()
    print("---------------------- Oversampling ----------------------")
    # pos_neg_ratio = len(df_train[df_train['Is Laundering']==1]) / len(df_train[df_train['Is Laundering']==0])
    # if pos_neg_ratio > 0:
    #     print("Length of training set:", len(df_train))
    #     OVERSAMPLING_RATIO = 0.2
    #     print("Positive negative ratio", pos_neg_ratio)

    #     while 1 - pos_neg_ratio > OVERSAMPLING_RATIO:
    #         df_train = pd.concat([df_train, df_train[df_train['Is Laundering']==1]], ignore_index=True)
    #         pos_neg_ratio = len(df_train[df_train['Is Laundering']==1]) / len(df_train[df_train['Is Laundering']==0])
        
    #     print("Length of training set after oversampling:", len(df_train))
    # else:
    #     print("Oversampling not needed because positive negative ratio is less than 0")
    print("---------------------- End oversampling ----------------------")
    # ----------------------------------------------------------------------------
  


    # SETTING UP DATASET
    # ----------------------------------------------------------------------------
    print()
    print("---------------------- Setting up dataset --------------------------")
    X_train, y_train = get_X_and_Y(df_train, verbose=VERBOSE)
    X_test, y_test = get_X_and_Y(df_test, verbose=VERBOSE)
    # print_dataset(X_train, y_train)
    X_train, _ = label_encoder(X_train, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
    X_test, _ = label_encoder(X_test, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
    
    # Iris dataset
    # encoder = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    # y_train = y_train.replace(encoder)
    # y_test = y_test.replace(encoder)
    # y_train[y_train == 2] = 0
    # y_test[y_test == 2] = 0
    print("---------------------- End setting up dataset --------------------------")
    # ----------------------------------------------------------------------------



    # PLOTTING
    # ----------------------------------------------------------------------------
    print()
    print("---------------------- Plotting --------------------------")
    df_train, df_train_label_decoder = label_encoder(df_train, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
    df_test, df_test_label_decoder = label_encoder(df_test, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
    plot_correlation_matrix(df_train) 
    plot_numerical_histograms(df_train)
    print("-------------------------- End plotting --------------------------")
    # ----------------------------------------------------------------------------
       


    # ID3
    # ----------------------------------------------------------------------------
    print()
    print("---------------------- ID3 --------------------------")    
    start_time = time.time()
    decision_tree: DecisionTreeID3 = DecisionTreeID3(max_depth=10, 
                                                     num_thresholds_numerical_attr=6)
    decision_tree.fit(X_train, y_train)
    end_time = time.time()
    decision_tree.create_dot_files(filename="tree-id3", generate_png=True, view=VIEW)
    print()
    print("Performances: ")
    predictions = list(decision_tree.predict_test(X_test))
    print(f"Fit time: {end_time - start_time} seconds") 
    calculate_performances(predictions, y_test, "id3", verbose=True)
    print("-------------------------- END ID3 --------------------------")
    # ----------------------------------------------------------------------------



    # CUSTOM
    # ----------------------------------------------------------------------------
    print()
    print("-------------------------- CUSTOM --------------------------")
    start_time = time.time()
    decision_tree = CustomDecisionTree(criterion=EntropyType.SHANNON, 
                                       type_criterion=CriterionType.BEST, 
                                       max_depth=10, 
                                       min_samples_split=14,
                                       num_thresholds_numerical_attr=6)
    decision_tree.fit(X_train, y_train)
    end_time = time.time()
    decision_tree.create_dot_files(filename="tree-custom", generate_png=True, view=VIEW)
    print()
    print("Performances: ") 
    predictions = list(decision_tree.predict_test(X_test))
    print(f"Fit time: {end_time - start_time} seconds")
    calculate_performances(predictions, y_test, "custom", verbose=True)
    print("-------------------------- END CUSTOM --------------------------")
    # ----------------------------------------------------------------------------
    
    assert(False)
    
    print()
    print("SKLEARN --------------------------")

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder
    """
    time_encoder = LabelEncoder()
    time_encoder.fit(df_train['Timestamp'])
    account_encoder = LabelEncoder()
    account_encoder.fit(df_train['Account'])
    account1_encoder = LabelEncoder()
    account1_encoder.fit(df_train['Account.1'])
    receiving_currency_encoder = LabelEncoder()
    receiving_currency_encoder.fit(df_train['Receiving Currency'])
    payment_currency_encoder = LabelEncoder()
    payment_currency_encoder.fit(df_train['Payment Currency'])
    payment_format_encoder = LabelEncoder()
    payment_format_encoder.fit(df_train['Payment Format'])

    df_train['Timestamp'] = time_encoder.transform(df_train['Timestamp'])
    df_train['Account'] = account_encoder.transform(df_train['Account'])
    df_train['Account.1'] = account1_encoder.transform(df_train['Account.1'])
    df_train['Receiving Currency'] = receiving_currency_encoder.transform(df_train['Receiving Currency'])
    df_train['Payment Currency'] = payment_currency_encoder.transform(df_train['Payment Currency'])
    df_train['Payment Format'] = payment_format_encoder.transform(df_train['Payment Format'])

    df_test['Timestamp'] = time_encoder.transform(df_test['Timestamp'])
    df_test['Account'] = account_encoder.transform(df_test['Account'])
    df_test['Account.1'] = account1_encoder.transform(df_test['Account.1'])
    df_test['Receiving Currency'] = receiving_currency_encoder.transform(df_test['Receiving Currency'])
    df_test['Payment Currency'] = payment_currency_encoder.transform(df_test['Payment Currency'])
    df_test['Payment Format'] = payment_format_encoder.transform(df_test['Payment Format'])

    X_train, y_train = get_X_and_Y(df_train, verbose=VERBOSE)
    X_test, y_test = get_X_and_Y(df_test, verbose=VERBOSE)"""

    X_train, _ = label_encoder(X_train, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
    X_test, _ = label_encoder(X_test, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])

    performances_accuracy = []
    performances_F1_score = []

    from sklearn import svm
    
    for i in range(n_iter):
    
        clf = DecisionTreeClassifier(splitter='best')
        # clf = svm.SVC()
        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        # import graphviz
        # import sklearn.tree as tree
        
        # graphviz.Source(tree.export_graphviz(clf, out_file="tree.dot", 
        #                                 class_names=['Yes', 'No'],
        #                                 feature_names=X_train.columns))
        
        #Predict the response for test dataset
        y_pred = clf.predict(X_test)

        accuracy, f1_score = calculate_performances(y_pred, y_test)

        performances_accuracy.append(accuracy)
        performances_F1_score.append(f1_score)

    print("\nAverage accuracy:", sum(performances_accuracy)/n_iter)
    print("Average F_1 score:", sum(performances_F1_score)/n_iter)
