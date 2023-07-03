import pandas as pd

def get_train_and_test(dataset_name = "LI-Small_Trans.csv", verbose = 0):
    df = pd.read_csv(f"datasets/{dataset_name}", sep=",")
    df_train = df.sample(frac=0.8, random_state=1)
    df_test  = df.drop(df_train.index)

    if verbose:
        print(df.shape)
        print(df.head())
        print(df["Is Laundering"].value_counts())
        
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

