import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

def get_train_and_test(dataset_name="", verbose = False, nrows = None):
    if nrows is None:
        df = pd.read_csv(f"datasets/{dataset_name}", sep=",")
    else:
        df = pd.read_csv(f"datasets/{dataset_name}", sep=",", nrows=nrows)
    df_train = df.sample(frac=0.8, random_state=2)
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

def split_timestamp(df: pd.DataFrame):
    # | Timestamp        |  -> | Date       | Hour | Minute  |
    # | ---------------- |     | ---------- | ---- | ------- |
    # | 2022/09/01 00:20 |  -> | 2022/09/01 | 00   | 20      |
    from datetime import datetime

    date_list, hour_list, minute_list = [], [], []

    # columns = list(df.columns[1:])

    for i in df.index:
        ts = df["Timestamp"][i]
        date_list.append(datetime.strptime(ts, "%Y/%m/%d  %H:%M").strftime("%Y/%m/%d"))
        hour_list.append(datetime.strptime(ts, "%Y/%m/%d  %H:%M").hour)
        minute_list.append(datetime.strptime(ts, "%Y/%m/%d  %H:%M").minute)

    df["Date"] = date_list
    df["Hour"] = hour_list
    df["Minute"] = minute_list

    df.drop(columns=["Timestamp"], inplace=True)

    # columns = ["Date", "Hour", "Minute"] + columns

    # df = df[columns]

    is_laundering = df.pop("Is Laundering")
    df.insert(len(df.columns), "Is Laundering", is_laundering)
