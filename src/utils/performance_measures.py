def calculate_performances(y_pred, y_test, verbose = False) -> (float, float):
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

    accuracy = (tp+tn)/len(y_test)
    f1_score = (2*tp)/(2*tp+fp+fn)

    if verbose :
        print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print("Accuracy:", accuracy)
        print("F1 score:", f1_score)
        # print("TP+TN:", tp+tn)
        # print("Length of test set:", len(y_test))


    return accuracy, f1_score

