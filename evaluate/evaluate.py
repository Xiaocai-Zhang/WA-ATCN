import numpy as np


def evaluation(test_pred,y_test):
    '''
    performance evaluation
    '''
    mae = np.mean(np.abs(test_pred - y_test))
    mse = np.mean(np.power(np.abs(test_pred - y_test), 2))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs(test_pred - y_test) / np.abs(y_test))

    return mae,rmse,mape
