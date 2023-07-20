# Calculate accuracy and F1 score
def calculate_performances(y_pred, y_test, verbose = False):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
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

    if verbose :
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print("TP+TN:", tp+tn)
        print("Length of test set:", len(y_test))

    accuracy = (tp+tn)/len(y_test)
    f1_score = (2*tp)/(2*tp+fp+fn)

    return accuracy, f1_score
