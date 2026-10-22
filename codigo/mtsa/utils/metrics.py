from sklearn import metrics

def calculate_aucroc(y_hat, y_test, pos_label=0):
    print(f"Calculating AUC-ROC with pos_label={pos_label}")
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_hat, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)
    return auc, fpr, tpr, thresholds

def calculate_confusion_matrix(y_hat, y_test):
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_hat).ravel()
    return tn, fp, fn, tp
