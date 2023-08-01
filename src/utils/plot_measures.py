import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(data, plot_size=(14, 12)):
    """
    Plot the correlation matrix of the dataset.
    """
    correlation_matrix = data.corr()
    plt.figure(figsize=plot_size)

    # Plot the correlation matrix with diagonal labels
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5,
                xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns,
                annot_kws={"size": 10})
    
    plt.xticks(rotation=45, ha='right')

    plt.title("Correlation Matrix")
    if plt.get_backend().lower() == 'module://ipykernel.pylab.backend_inline':
        plt.show()
    else:
        figs_directory = "plt_figs" 
        if not os.path.exists(figs_directory):
            os.makedirs(figs_directory)

        plt.savefig(os.path.join(figs_directory, "correlation_matrix.png"))
        plt.close()

def plot_numerical_histograms(data, plot_size=(14, 12)):
    """
    Plot histograms for each numerical column in the dataset.
    """
    numerical_columns = data.select_dtypes(include=[np.number])
    numerical_columns.hist(bins=20, figsize=plot_size)
    plt.suptitle("Histograms of Numerical Columns", y=1.02)
    plt.tight_layout()
    
    if plt.get_backend().lower() == 'module://ipykernel.pylab.backend_inline':
        plt.show()
    else:
        figs_directory = "plt_figs"
        if not os.path.exists(figs_directory):
            os.makedirs(figs_directory)

        plt.savefig(os.path.join(figs_directory, "numerical_histograms.png"))
        plt.close()

def plot_confusion_matrix(cm, classes, model_name, normalize=False, plot_size=(16.5, 14)):
    """
    Plot the confusion matrix for the model.
    """
    plt.figure(figsize=plot_size)

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.

    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    if plt.get_backend().lower() == 'module://ipykernel.pylab.backend_inline':
        plt.show()
    else:
        figs_directory = "plt_figs"
        if not os.path.exists(figs_directory):
            os.makedirs(figs_directory)

        plt.savefig(os.path.join(figs_directory, f"confusion_matrix_{model_name}.png"))
        plt.close()

def plot_roc_curve(fpr, tpr, auc, model_name, plot_size=(14, 12)):
    """
    Plot the ROC curve for the model.
    """
    plt.figure(figsize=plot_size)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    
    if plt.get_backend().lower() == 'module://ipykernel.pylab.backend_inline':
        plt.show()
    else:
        figs_directory = "plt_figs"
        if not os.path.exists(figs_directory):
            os.makedirs(figs_directory)

        plt.savefig(os.path.join(figs_directory, f"roc_curve_{model_name}.png"))
        plt.close()
