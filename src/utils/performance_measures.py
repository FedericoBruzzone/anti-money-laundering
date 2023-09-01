
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

def accuracy(tp, tn, n_y) -> float:
    return (tp+tn)/n_y

def precision(tp, fp) -> float:
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def recall(tp, fn) -> float:
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def f1_score(tp, fn, fp) -> float:
    p = precision(tp, fp) 
    r = recall(tp, fn)
    if p + r == 0:
        return 0
    return 2 * (p * r) / (p + r)

def calculate_performances(y_pred, y_test, model_name="", verbose = False) -> (float, float):
    tp, tn, fp, fn = confusion_matrix(y_pred, y_test)
    acc            = accuracy(tp, tn, len(y_test))
    f1_s           = f1_score(tp, fn, fp)
    tpr            = tp/(tp+fn) if (tp+fn) != 0 else 0
    fpr            = fp/(fp+tn) if (fp+tn) != 0 else 0

    if verbose :
        print("%12s: %7.6f %12s: %7.6f" % ("F1 score", f1_s, "Accuracy", acc))
        print("%12s: %7.6f %12s: %7.6f" % ("Precision", precision(tp, fp), "Recall", recall(tp, fn)))
        
        print("%12s: %8d %12s: %8d" % ("TP", tp, "TN", tn))
        print("%12s: %8d %12s: %8d" % ("FP", fp, "FN", fn))
        print("%12s: %7.6f %12s: %7.6f" % ("TPR", tpr, "FPR", fpr))

    # tpr_list, fpr_list = roc_curve(y_pred, y_test)
    # cm = np.array([[tp, fp], [fn, tn]])
    # classes = ['0', '1']
    # plot_roc_curve(fpr_list, tpr_list, 1, model_name)
    # plot_confusion_matrix(cm, classes, model_name, normalize=False)

    return accuracy, f1_score

