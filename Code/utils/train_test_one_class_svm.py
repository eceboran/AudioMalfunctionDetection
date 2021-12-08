import numpy as np
from sklearn.svm import OneClassSVM
from sklearn import metrics
from utils.confusion_metrics import confusion_metrics

def train_test_one_class_svm(X_train, X_test, y_train, y_test, kernel='rbf', gamma=0.001, nu=0.5, prctle=2, normalize="min-max"):
    
    # Normalization
    if normalize=="min-max":
        for no_feat in range(X_train.shape[1]):
            minval = np.min(X_train[:,no_feat])
            maxval = np.max(X_train[:,no_feat])

            X_train[:,no_feat] = (X_train[:,no_feat]-minval)/(maxval-minval)
            X_test[:,no_feat] = (X_test[:,no_feat]-minval)/(maxval-minval)
    
    
    svm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)  # create classifier

    svm.fit(X_train)  # fit training data
    
    y_train_pred = svm.predict(X_train)  # predict training targets
    y_train_pred = np.array([1 if elem==-1 else 0 for elem in y_train_pred]).reshape(-1, 1)
    scores_training = svm.score_samples(X_train)  # scores for training data

    # Threshold from scores of training set
    thresh = np.percentile(scores_training, prctle)
    
    # Training classes predicted with score
    y_train_pred_score = np.array([1 if elem<thresh else 0 for elem in scores_training]).reshape(-1, 1)

    # Test set
    y_test_pred = svm.predict(X_test)
    y_test_pred = np.array([1 if elem==-1 else 0 for elem in y_test_pred]).reshape(-1, 1)
    scores_test = svm.score_samples(X_test)
    
    # Test classes predicted with score
    y_test_pred_score = np.array([1 if elem<thresh else 0 for elem in scores_test]).reshape(-1, 1)

    report = metrics.classification_report(y_test, y_test_pred_score)

    cm_test = metrics.confusion_matrix(y_test, y_test_pred_score)
    cm_train = metrics.confusion_matrix(y_train, y_train_pred_score)

    out_metrics_test = confusion_metrics(cm_test)
    out_metrics_train = confusion_metrics(cm_train)
    
    params = {'kernel': kernel,
             'gamma': gamma,
             'nu': nu,
             'threshold': thresh,
             'prctle': prctle,
             'normalize': normalize}
    
    return OneClassSVM, report, cm_train, cm_test, out_metrics_train, out_metrics_test, params