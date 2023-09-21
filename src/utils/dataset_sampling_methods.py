import pandas as pd

def oversampling(df_train, VERBOSE):
    if VERBOSE:
        print()
        print("---------------------- Oversampling ----------------------")
    pos_neg_ratio = len(df_train[df_train['Is Laundering']==1]) / len(df_train[df_train['Is Laundering']==0])
    if pos_neg_ratio > 0:
        OVERSAMPLING_RATIO = 0.5
        if VERBOSE:
            print("Length of training set:", len(df_train))
            print("Positive negative ratio", pos_neg_ratio)

        while 1 - pos_neg_ratio > OVERSAMPLING_RATIO:
            df_train = pd.concat([df_train, df_train[df_train['Is Laundering']==1]], ignore_index=True)
            pos_neg_ratio = len(df_train[df_train['Is Laundering']==1]) / len(df_train[df_train['Is Laundering']==0])

        if VERBOSE:
            print("Length of training set after oversampling:", len(df_train))
    else:
        if VERBOSE:
            print("Oversampling not needed because positive negative ratio is less than 0")
    if VERBOSE:
        print("---------------------- End oversampling ----------------------")
    return df_train

def undersampling(df_train, VERBOSE):
    if VERBOSE:
        print()
        print("---------------------- Undersampling ----------------------")
        print("Length of training set:", len(df_train))
    is_laundering = df_train[df_train['Is Laundering']==1]
    is_not_laundering = df_train[df_train['Is Laundering']==0]
    is_not_laundering = is_not_laundering.sample(n=len(is_laundering), random_state=101)
    df_train = pd.concat([is_laundering, is_not_laundering],axis=0)
    if VERBOSE:
        print("Length of training set after undersampling:", len(df_train))
        print("---------------------- End undersampling ----------------------")
    return df_train

def bootstrap_sampling(df_train, VERBOSE=False):
    if VERBOSE:
        print()
        print("---------------------- Bootstrap sampling ----------------------")
        print("Length of training set:", len(df_train))
    df_train = df_train.sample(n=len(df_train), replace=True)
    if VERBOSE:
        print("Length of training set after bootstrap sampling:", len(df_train))
        print("---------------------- End bootstrap sampling ----------------------")
    return df_train

