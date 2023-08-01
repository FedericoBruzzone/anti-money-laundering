import numpy as np
from src.utils.plot_measures import plot_roc_curve
from src.utils.plot_measures import plot_confusion_matrix

def confusion_matrix(y_pred, y_test) -> (int, int, int, int):
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    for i, p in enumerate(y_pred):
        if p == y_test.iloc[i]:
            if p == 1:
                tp += 1
            else:
                tn += 1
        else:
            if p == 1:
                fp += 1
            else:
                fn += 1
    return tp, tn, fp, fn

def roc_curve(y_pred, y_test):
    tpr_list = [0.0]
    fpr_list = [0.0]
    
    thresholds = sorted(set(y_pred), reverse=True)
    for threshold in thresholds:
        y_pred_thresholded = [1 if p >= threshold else 0 for p in y_pred]
        tp, tn, fp, fn = confusion_matrix(y_pred_thresholded, y_test)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)    
    return tpr_list, fpr_list

def accuracy(y_pred, y_test) -> float:
    tp, tn, _, _ = confusion_matrix(y_pred, y_test)
    return (tp+tn)/len(y_test)

def precision(y_pred, y_test) -> float:
    tp, _, fp, _ = confusion_matrix(y_pred, y_test)
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def recall(y_pred, y_test) -> float:
    tp, _, _, fn = confusion_matrix(y_pred, y_test)
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def f1_score(y_pred, y_test) -> float:
    p = precision(y_pred, y_test) 
    r = recall(y_pred, y_test)
    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)

def calculate_performances(y_pred, y_test, model_name, verbose = False) -> (float, float):
    tp, tn, fp, fn = confusion_matrix(y_pred, y_test)
    acc            = accuracy(y_pred, y_test)
    f1_s           = f1_score(y_pred, y_test)
    tpr            = tp/(tp+fn) if (tp+fn) != 0 else 0
    fpr            = fp/(fp+tn) if (fp+tn) != 0 else 0

    tpr_list, fpr_list = roc_curve(y_pred, y_test)
    cm = np.array([[tp, fp], [fn, tn]])
    classes = ['0', '1']

    if verbose :
        print("F1 score:", f1_s)
        print("Accuracy:", acc)
        print("Precision:", precision(y_pred, y_test))
        print("Recall:", recall(y_pred, y_test))
        
        print("True positive: ", tp)
        print("True negative: ", tn)
        print("False positive: ", fp)
        print("False negative: ", fn)
        print("True positive rate:", tpr)
        print("False positive rate:", fpr)
    
    plot_roc_curve(fpr_list, tpr_list, 1, model_name)
    plot_confusion_matrix(cm, classes, model_name, normalize=False) 

    return accuracy, f1_score

