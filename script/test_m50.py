# -*- coding: utf-8 -*-
import pandas as pd
pd.options.mode.chained_assignment = None
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from processing.M50.processing import processing
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
from evaluate.evaluate import evaluation
import numpy as np
import random



class config:
    '''
    config parameter and path
    '''
    n_classes = 1
    TrainValRatio = [0.8, 0.2]
    date_start = '2018-03-06'
    date_split = '2019-03-02'
    data_path = './data/M50/mean_flow.csv'
    ParaPath = './para/parameter_m50.csv'
    SavePath = './para/test_result_m50.csv'
    seed = 10


# set seed
np.random.seed(config.seed)
random.seed(config.seed)
os.environ['PYTHONHASHSEED']=str(config.seed)
tf.random.set_seed(config.seed)


def dataset(timestep):
    '''
    process data sets
    '''
    flow_data = pd.read_csv(config.data_path)
    x_train,y_train,x_test,y_test = processing(flow_data, timestep, config.date_start, config.date_split, train=False).main()

    # split to train, val sets
    num_val = int(len(x_train) * config.TrainValRatio[1])

    IdxTrainVal = [idx for idx in range(0, x_train.shape[0])]

    IdxVal = random.sample(IdxTrainVal, num_val)
    IdxTrain = [idx for idx in IdxTrainVal if idx not in IdxVal]

    x_train_ = x_train[IdxTrain, :, :]
    y_train_ = y_train[IdxTrain, :]
    x_val_ = x_train[IdxVal, :, :]
    y_val_ = y_train[IdxVal, :]

    return x_train_,y_train_,x_val_,y_val_,x_test,y_test


class attention(Layer):
    '''
    define attention mechanism
    '''
    def __init__(self, return_sequences=True,**kwargs):
        self.return_sequences = return_sequences
        super(attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_sequences': self.return_sequences
        })
        return config


def Test(x,round_no):
    '''
    test performance on test data
    '''
    timestep = int(round(x[0]))
    batchsize = int(round(x[1]))

    LastIdx = [idx for idx in range(50)]
    ValdIdx = LastIdx[-timestep:]

    x_train_loc = x_train_glo[:, ValdIdx, :]
    x_test_loc = x_test_glo[:, ValdIdx, :]

    min_flow = x_train_loc[:, :, 0].min()
    max_flow = x_train_loc[:, :, 0].max()

    # data Normilization
    x_test_loc[:, :, 0] = (x_test_loc[:, :, 0] - min_flow) / (max_flow - min_flow)

    string = ""
    for i in range(len(x)):
        if i in [0, 1, 2, 3, 4, 8]:
            item = int(round(x[i]))
        else:
            item = round(x[i], 4)
        string = string + '_' + str(item)

    SaveModlPath = './save/M50/round_' + str(round_no) + '/'
    SaveModlFile = SaveModlPath + 'model' + string + '.h5'

    # load model
    model = load_model(SaveModlFile,custom_objects={'attention': attention})

    predictions_test = model.predict(x_test_loc, batch_size=batchsize)
    np.savetxt('./prediction/M50/prediction.csv',predictions_test,delimiter=',')
    np.savetxt('./prediction/M50/groundtruth.csv', y_test_glo, delimiter=',')
    mae_test, rmse_test, mape_test = evaluation(predictions_test, y_test_glo)

    return mae_test, rmse_test, mape_test


if __name__ == "__main__":
    global x_train_glo, y_train_glo, x_test_glo, y_test_glo
    x_train_glo, y_train_glo, _, _, x_test_glo, y_test_glo = dataset(50)

    # read saved best hyper-parameters
    ParaTable = pd.read_csv(config.ParaPath, dtype='string')

    mae_li = []
    rmse_li = []
    mape_li = []
    time_li = []
    for index, row in ParaTable.iterrows():
        round_no = int(row['round_no'])
        time = float(row['time'])
        Para = row['best_para']
        Para_r = Para.replace("[", "").replace("]", "")
        Para_s = Para_r.split(", ")
        Para_f = [float(item) for item in Para_s]

        mae_test, rmse_test, mape_test = Test(Para_f, round_no)
        print('MAE: %s, RMSE: %s, MAPE: %s' % (mae_test, rmse_test, mape_test))
        mae_li.append(mae_test)
        rmse_li.append(rmse_test)
        mape_li.append(mape_test)
        time_li.append(time)

    SaveDf = pd.DataFrame({'MAE': mae_li, 'RMSE': rmse_li, 'MAPE': mape_li, 'Time': time_li})
    SaveDf.to_csv(config.SavePath, index=False)