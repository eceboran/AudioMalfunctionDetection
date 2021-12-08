import numpy as np
import pandas as pd


def get_train_test_data(X, y, df):

    if type(X) == pd.core.frame.DataFrame:
        X_train = X[df.test_train == 0]
        X_test = X[df.test_train == 1]
        y_train = y[df.test_train == 0]
        y_test = y[df.test_train == 1]

    elif type(X) == np.ndarray:
        X_train = X[(df.test_train==0).squeeze(), :]
        X_test = X[(df.test_train==1).squeeze(), :]
        y_train = y[(df.test_train==0).squeeze(), :]
        y_test = y[(df.test_train==1).squeeze(), :]


    return X_train, X_test, y_train, y_test