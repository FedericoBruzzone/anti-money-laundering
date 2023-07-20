import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from src.kaggle_config import setup_kaggle
from src.kaggle_config import download_dataset
from src.utils.datasets_handler import get_train_and_test
from src.utils.datasets_handler import get_X_and_Y
from src.utils.datasets_handler import print_dataset
from src.utils.datasets_handler import label_encoder
from src.decision_tree.decision_tree import DecisionTree
import src.utils.performance_measures as p_measures

if __name__ == "__main__":
    VERBOSE = int(os.getenv('VERBOSE'))

    setup_kaggle()
    print("Downloading dataset...")
    download_dataset()
    print("Done!")

    df_train, df_test = get_train_and_test(verbose=VERBOSE)


    # Undersampling --------
    # print("df len: ", len(df_train))
    # is_laundering = df_train[df_train['Is Laundering']==1]
    # is_not_laundering = df_train[df_train['Is Laundering']==0]
    # is_not_laundering = is_not_laundering.sample(n=len(is_laundering), random_state=101)
    # df_train = pd.concat([is_laundering, is_not_laundering],axis=0)
    # print("Undersampled df len: ", len(df_train))
    # ----------------------
    

    # Oversampling --------
    pos_neg_ratio = len(df_train[df_train['Is Laundering']==1]) / len(df_train[df_train['Is Laundering']==0])
    print("RATIO", pos_neg_ratio)

    while 1 - pos_neg_ratio > 0.1:
        print("Oversampling...", 1 - pos_neg_ratio)

        df_train = pd.concat([df_train, df_train[df_train['Is Laundering']==1]], ignore_index=True)
        pos_neg_ratio = len(df_train[df_train['Is Laundering']==1]) / len(df_train[df_train['Is Laundering']==0])
    
    print("Length of training set:", len(df_train))

    # ----------------------

    # scikit-learn
    X_train, y_train = get_X_and_Y(df_train, verbose=VERBOSE)
    X_test, y_test = get_X_and_Y(df_test, verbose=VERBOSE)
    X_train, _ = label_encoder(X_train, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
    X_test, _ = label_encoder(X_test, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
    # print_dataset(X_train, y_train)


    performances_accuracy = []
    performances_F1_score = []

    n_iter = 5

    for i in range(n_iter):
        print("Iteration:", i)
    
        decision_tree: DecisionTree = DecisionTree(type_criterion=0, max_depth=6)
        decision_tree.fit(X_train, y_train)
        print(decision_tree)
        decision_tree.create_dot_files(generate_png=True, view=False)


        predictions = list(decision_tree.predict_test(X_test))
        
        accuracy, f1_score = p_measures.calculate_performances(predictions, y_test)
        
        print("Accuracy:", accuracy)
        print("F_1 score:", f1_score)

        performances_accuracy.append(accuracy)
        performances_F1_score.append(f1_score)
    
    print("\n\nAverage accuracy:", sum(performances_accuracy)/n_iter)
    print("Average F_1 score:", sum(performances_F1_score)/n_iter)

    # print(type(X_train.iloc[0].values)) # <class 'numpy.ndarray'>
    # print(X_train.iloc[0]["Timestamp"])

    
    # Scikit-learn
    print("\n\nSKLEARN --------------------------")

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder

    from sklearn.preprocessing import LabelEncoder

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

    performances_accuracy = []
    performances_F1_score = []
    
    for i in range(n_iter):
    
        clf = DecisionTreeClassifier(splitter='random')

        # Train Decision Tree Classifer
        clf = clf.fit(X_train,y_train)

        # import graphviz
        # import sklearn.tree as tree
        
        # graphviz.Source(tree.export_graphviz(clf, out_file="tree.dot", 
        #                                 class_names=['Yes', 'No'],
        #                                 feature_names=X_train.columns))
        
        #Predict the response for test dataset
        y_pred = clf.predict(X_test)

        accuracy, f1_score = p_measures.calculate_performances(y_pred, y_test)

        performances_accuracy.append(accuracy)
        performances_F1_score.append(f1_score)

    print("\nAverage accuracy:", sum(performances_accuracy)/n_iter)
    print("Average F_1 score:", sum(performances_F1_score)/n_iter)