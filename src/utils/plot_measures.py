import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(data):
    """
    Plot the correlation matrix of the dataset.
    """
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.show()

# plot histogram for categorical columns
def plot_categorical_histograms(data):
    """
    Plot histograms for each categorical column in the dataset.
    """
    categorical_columns = data.select_dtypes(include=[np.object])
    categorical_columns.hist(figsize=(12, 10))
    plt.suptitle("Histograms of Categorical Columns", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_numerical_histograms(data):
    """
    Plot histograms for each numerical column in the dataset.
    """
    numerical_columns = data.select_dtypes(include=[np.number])
    numerical_columns.hist(bins=20, figsize=(12, 10))
    plt.suptitle("Histograms of Numerical Columns", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_categorical_scatterplots(data):
    """
    Plot scatterplots for each categorical column in the dataset.
    """
    categorical_columns = data.select_dtypes(include=[np.object])
    sns.pairplot(categorical_columns)
    plt.suptitle("Scatterplots of Categorical Columns", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_numerical_scatterplots(data):
    """
    Plot scatterplots for each numerical column in the dataset.
    """
    numerical_columns = data.select_dtypes(include=[np.number])
    sns.pairplot(numerical_columns)
    plt.suptitle("Scatterplots of Numerical Columns", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_categorical_boxplots(data):
    """
    Plot boxplots for each categorical column in the dataset.
    """
    categorical_columns = data.select_dtypes(include=[np.object])
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=categorical_columns, orient='h')
    plt.title("Boxplots of Categorical Columns")
    plt.show()

def plot_numerical_boxplots(data):
    """
    Plot boxplots for each numerical column in the dataset.
    """
    numerical_columns = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=numerical_columns, orient='h')
    plt.title("Boxplots of Numerical Columns")
    plt.show()

def plot_categorical_distribution(data, column):
    """
    Plot the distribution of a categorical column in the dataset.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    plt.figure(figsize=(8, 6))
    sns.countplot(x=column, data=data)
    plt.title(f"Distribution of '{column}'")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_numerical_distribution(data, column):
    """
    Plot the distribution of a numerical column in the dataset.
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f"Distribution of '{column}'")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

