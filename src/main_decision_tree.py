import os
from dotenv import load_dotenv
load_dotenv()

import time
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
from src.decision_tree.decision_tree import CustomDecisionTree
import src.utils.performance_measures as p_measures

if __name__ == "__main__":
    VERBOSE = int(os.getenv('VERBOSE'))

    setup_kaggle()
    print("Downloading dataset...")
    download_dataset()
    print("Done!")

    df_train, df_test = get_train_and_test("HI-Small_Trans.csv", verbose=VERBOSE)

    # df_train2, df_test2 = get_train_and_test("HI-Small_Trans.csv", verbose=VERBOSE)

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

    # pos_neg_ratio2 = len(df_train2[df_train2['Is Laundering']==1]) / len(df_train2[df_train2['Is Laundering']==0])
    # print("RATIO_HIGH", pos_neg_ratio2)

    while 1 - pos_neg_ratio > 0.1:
        # print("Oversampling...", 1 - pos_neg_ratio)

        df_train = pd.concat([df_train, df_train[df_train['Is Laundering']==1]], ignore_index=True)
        pos_neg_ratio = len(df_train[df_train['Is Laundering']==1]) / len(df_train[df_train['Is Laundering']==0])
    
    print("Length of training set:", len(df_train))

    # ----------------------

    X_train, y_train = get_X_and_Y(df_train, verbose=VERBOSE)
    X_test, y_test = get_X_and_Y(df_test, verbose=VERBOSE)
    
    # print_dataset(X_train, y_train)

    # TEST ID3
    from src.decision_tree.ID3 import DecisionTreeID3

    X_train["Account"] = X_train["Account"].apply(lambda x: int(str(x), 16))
    X_test["Account"] = X_test["Account"].apply(lambda x: int(str(x), 16))

    X_train["Account.1"] = X_train["Account"].apply(lambda x: int(str(x), 16))
    X_test["Account.1"] = X_test["Account"].apply(lambda x: int(str(x), 16))

    #X_train.astype({'Account': 'int32'})

    start_time = time.time()

    decision_tree: DecisionTreeID3 = DecisionTreeID3(continuous_attr_groups=8)
    decision_tree.fit(X_train, y_train)

    print("--- Fit time: %s seconds ---" % (time.time() - start_time))

    print(decision_tree)
    decision_tree.create_dot_files(generate_png=True, view=False)

    #predictions = list(decision_tree.predict_test(X_test))

    tp, tn, fp, fn = 0, 0, 0, 0
    try:
        for index, row in X_test.iterrows():
            pred = decision_tree.predict(row)
            if pred == y_test[index]:
                if pred:
                    tp += 1
                else:
                    tn += 1
            else:
                if pred:
                    fp += 1
                else:
                    fn += 1
    except:
        pass

    print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)

    accuracy_id3 = (tp+tn)/len(y_test)
    f1_score_id3 = (2*tp)/(2*tp+fp+fn)

    print("Accuracy:", accuracy_id3)
    print("F_1 score:", f1_score_id3)

    assert(False)

    # accuracy, f1_score = p_measures.calculate_performances(predictions, y_test)
        
    # print("Accuracy:", accuracy)
    # print("F_1 score:", f1_score)    

    ##############################

    X_train, _ = label_encoder(X_train, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])
    X_test, _ = label_encoder(X_test, ['Timestamp', 'Account', 'Account.1', 'Receiving Currency', 'Payment Currency', 'Payment Format'])

    performances_accuracy = []
    performances_F1_score = []

    n_iter = 1

    print("\nCUSTOM --------------------------")

    for i in range(n_iter):
        
        print("Iteration:", i)
    
        decision_tree = CustomDecisionTree(criterion=0, type_criterion=0, max_depth=10)
        decision_tree.fit(X_train, y_train)
        print(decision_tree)
        decision_tree.create_dot_files(generate_png=True, view=False)


        predictions = list(decision_tree.predict_test(X_test))
        
        accuracy, f1_score = p_measures.calculate_performances(predictions, y_test)
        
        print("Accuracy:", accuracy)
        print("F_1 score:", f1_score)

        performances_accuracy.append(accuracy)
        performances_F1_score.append(f1_score)
    
    # print("\n\nAverage accuracy:", sum(performances_accuracy)/n_iter)
    # print("Average F_1 score:", sum(performances_F1_score)/n_iter)

    # print(type(X_train.iloc[0].values)) # <class 'numpy.ndarray'>
    # print(X_train.iloc[0]["Timestamp"])

    
    # Scikit-learn
    print("\n\nSKLEARN --------------------------")

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

        accuracy, f1_score = p_measures.calculate_performances(y_pred, y_test)

        performances_accuracy.append(accuracy)
        performances_F1_score.append(f1_score)

    print("\nAverage accuracy:", sum(performances_accuracy)/n_iter)
    print("Average F_1 score:", sum(performances_F1_score)/n_iter)