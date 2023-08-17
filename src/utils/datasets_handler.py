import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

def get_train_and_test(dataset_name="", verbose = False):
    df = pd.read_csv(f"datasets/{dataset_name}", sep=",") #, nrows=300000)
    df_train = df.sample(frac=0.7, random_state=1)
    df_test  = df.drop(df_train.index)

    # print(df_train.head(5))
    # print(df_train.nunique())

    if verbose:
        print(df.shape)
        print(df.head())
        print(df.iloc[:, -1].value_counts())
        
    return df_train, df_test

def get_X_and_Y(df, verbose = 0):
    X_df = df.iloc[:, :-1]
    Y_df = df.iloc[:, -1]
    
    if verbose:
        print(X_df.shape)
        print(X_df.head())
        print(Y_df.shape)
        print(Y_df.head())

    return (X_df, Y_df)

def label_encoder(df: pd.DataFrame, columns: list[str]):
        encoders = {}
        for column in columns:
            unique_values = df[column].unique()
            encoder = {value: np.int64(index) for index, value in enumerate(unique_values)}
            df[column] = df[column].map(encoder)
            encoders[column] = encoder
        return df, encoders

def print_dataset(X_train, Y_train):
    for i in range(len(X_train)):
        print(X_train.iloc[i])
        print(Y_train.iloc[i])
        print()
