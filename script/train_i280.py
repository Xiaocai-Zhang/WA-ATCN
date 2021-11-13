import pandas as pd
pd.options.mode.chained_assignment = None
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from processing.I280.processing import processing
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
from evaluate.evaluate import evaluation
from de.differential_evolution import DEAlgorithm
import pathlib
import shutil
import numpy as np
import random
from datetime import datetime
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class config:
    '''
    config parameter and path
    '''
    n_classes = 1
    TrainValRatio = [0.8, 0.2]
    date_start = '2018-04-01'
    date_split = '2019-01-17'
    data_path = './data/I280/mean_flow.csv'
    ParaSavePath = './para/parameter_i280.csv'
    stopmarginloss = 3*10**4

    # timestep, batchsize, channel, layer, kernel, dropout, lr, lr_factor, patience
    bounds_all = [(1, 50),(16, 512),(1, 50),(1, 10),(1, 50),(0, 1),(0.01, 0.1),(0,1),(1,10)]
    F_c = 0.7
    EarlyStopStep = 3
    maxiter = 10
    F_list = [(0.8,1),(0.6,0.8),(0.4,0.6),(0.2,0.4),(0,0.2),(0,0.2),(0,0.2),(0,0.2),(0,0.2),(0,0.2)]
    k_list = [50, 30, 20, 10, 5, 5, 5, 5, 5, 5]  # length of popsize must be equal to maxiter
    seed = 11


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
    x_train,y_train = processing(flow_data, timestep, config.date_start, config.date_split, train=True).main()

    # split to train, val sets
    num_val = int(len(x_train) * config.TrainValRatio[1])

    IdxTrainVal = [idx for idx in range(0, x_train.shape[0])]

    IdxVal = random.sample(IdxTrainVal, num_val)
    IdxTrain = [idx for idx in IdxTrainVal if idx not in IdxVal]

    x_train_ = x_train[IdxTrain, :, :]
    y_train_ = y_train[IdxTrain, :]
    x_val_ = x_train[IdxVal, :, :]
    y_val_ = y_train[IdxVal, :]

    return x_train_,y_train_,x_val_,y_val_


class TempoConvNetworks:
    def TcnBlock(self, x, dilation_rate, nb_filters, kernel_size, dropout, padding):
        '''
        define TCN block
        '''
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']
        conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                       kernel_initializer=init)
        batch1 = BatchNormalization(axis=-1)
        ac1 = Activation('relu')
        drop1 = GaussianDropout(dropout)

        conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                       kernel_initializer=init)
        batch2 = BatchNormalization(axis=-1)
        ac2 = Activation('relu')
        drop2 = GaussianDropout(dropout)

        downsample = Conv1D(filters=nb_filters, kernel_size=1, padding='same', kernel_initializer=init)
        ac3 = Activation('relu')

        pre_x = x

        x = conv1(x)
        x = batch1(x)
        x = ac1(x)
        x = drop1(x)
        x = conv2(x)
        x = batch2(x)
        x = ac2(x)
        x = drop2(x)

        if pre_x.shape[-1] != x.shape[-1]:  # to match the dimensions
            pre_x = downsample(pre_x)

        assert pre_x.shape[-1] == x.shape[-1]

        out = ac3(pre_x + x)

        return out

    def TcnNet(self, input, num_channels, kernel_size, dropout):
        '''
        define TCN network
        '''
        assert isinstance(num_channels, list)
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_rate = 2 ** i
            input = self.TcnBlock(input, dilation_rate, num_channels[i], kernel_size, dropout, padding='causal')

        out = input

        return out


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


class CustomCallback(tf.keras.callbacks.Callback):
    '''
    custom callback function
    '''
    def __init__(self, Global_fir_epoch_loss, config):
        super(tf.keras.callbacks.Callback, self).__init__()
        self.Global_fir_epoch_loss = Global_fir_epoch_loss
        self.config = config

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')
        if val_loss>self.Global_fir_epoch_loss+config.stopmarginloss:
            self.model.stop_training = True


def Train(x, round_no, Global_fir_epoch_loss, f, num, step):
    '''
    train ATCN network
    '''
    timestep = int(round(x[0]))
    batchsize = int(round(x[1]))
    channel = int(round(x[2]))
    layer = int(round(x[3]))
    kernel_tcn = int(round(x[4]))
    dropout = round(x[5], 4)
    lr = round(x[6], 4)
    lr_f = round(x[7], 4)
    patience = int(round(x[8]))
    num_channels = [channel] * layer

    LastIdx = [idx for idx in range(50)]
    ValdIdx = LastIdx[-timestep:]

    x_train_loc = x_train_glo[:, ValdIdx, :]
    x_val_loc = x_val_glo[:, ValdIdx, :]

    min_flow = x_train_loc[:, :, 0].min()
    max_flow = x_train_loc[:, :, 0].max()

    # data Normilization
    x_train_loc[:, :, 0] = (x_train_loc[:, :, 0] - min_flow) / (max_flow - min_flow)
    x_val_loc[:, :, 0] = (x_val_loc[:, :, 0] - min_flow) / (max_flow - min_flow)

    inp_shape = (timestep, 5)
    input = Input(shape=inp_shape)

    # TCN block
    TCNetworks = TempoConvNetworks()
    output = TCNetworks.TcnNet(input, num_channels, kernel_tcn, dropout)

    output = attention(return_sequences=False)(output)

    # desne layer
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    output = Dense(config.n_classes,kernel_initializer=init)(output)
    output = output * (max_flow - min_flow) + min_flow

    model = Model(inputs=input, outputs=output)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt)

    # shuffle training data
    np.random.seed(config.seed)
    x_train_loc = np.random.permutation(x_train_loc)
    np.random.seed(config.seed)
    y_train_loc = np.random.permutation(y_train_glo)

    string = ""
    for i in range(len(x)):
        if i in [0, 1, 2, 3, 4, 8]:
            item = int(round(x[i]))
        else:
            item = round(x[i], 4)
        string = string + '_' + str(item)

    SaveModlPath = './save/I280/round_' + str(round_no) + '/'
    SaveModlFile = SaveModlPath + 'model' + string + '.h5'
    pathlib.Path(SaveModlPath).mkdir(parents=True, exist_ok=True)

    # define callbacks
    cus_callback = CustomCallback(Global_fir_epoch_loss,config)
    mcp_save = callbacks.ModelCheckpoint(SaveModlFile, save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_f, patience=patience, min_lr=0.001, mode='min')

    if num==0 and step==0:
        history = model.fit(x=x_train_loc, y=y_train_loc, epochs=150,
                            batch_size=batchsize, validation_data=(x_val_loc, y_val_glo),
                            callbacks=[mcp_save, early_stopping, reduce_lr_loss],
                            verbose=0)
    else:
        history = model.fit(x=x_train_loc, y=y_train_loc, epochs=150,
                  batch_size=batchsize, validation_data=(x_val_loc, y_val_glo),
                  callbacks=[mcp_save,early_stopping,reduce_lr_loss,cus_callback],
                  verbose=0)

    fir_epoch_loss = history.history["val_loss"][0]

    # load model
    try:
        model = load_model(SaveModlFile,custom_objects={'attention': attention})
    except:
        pass

    predictions_val = model.predict(x_val_loc, batch_size=batchsize)
    _, _, mape_val = evaluation(predictions_val, y_val_glo)

    K.clear_session()
    del model

    printtxt = "MAPE: %.4f: %s,%s,%s,%s,%s,%s,%s,%s,%s" % (
    mape_val, timestep, batchsize, channel, layer, kernel_tcn, dropout, lr, lr_f, patience)
    print(printtxt)
    os.write(f, str.encode(printtxt + '\n'))

    return mape_val,fir_epoch_loss


def SelectTopK(InitialArray,BestMape,step,k):
    '''
    function to select the best-k candidates
    '''
    if step == 0:
        return InitialArray,BestMape
    else:
        topkidx = sorted(range(len(BestMape)), key=lambda i: BestMape[i])[0:k]
        topkidx.sort()
        topkarray = [InitialArray[i] for i in topkidx]
        BestMape_ = [BestMape[i] for i in topkidx]

        return topkarray,BestMape_


def RunDE(round_no):
    '''
    ABDE algorithm
    '''
    start = datetime.now()

    DEAlgorithmClass = DEAlgorithm(config)
    initial_popusize = config.k_list[0]
    InitialArray = DEAlgorithmClass.initialization(initial_popusize)

    GlobalMinValLi = []
    BestMape = []
    GlobalMinVal = 1000

    SaveTextPath = './result/I280/round_' + str(round_no) + '/'
    shutil.rmtree(SaveTextPath)
    pathlib.Path(SaveTextPath).mkdir(parents=True, exist_ok=True)
    f = os.open(SaveTextPath + "prcoess.txt", os.O_RDWR | os.O_CREAT)

    Global_fir_epoch_loss = 0
    Global_step_mape = 1000
    for step in range(config.maxiter):
        printtxt = 'Round: %s, Step %s' % (round_no,step)
        print(printtxt)
        os.write(f, str.encode(printtxt + "\n"))

        # select top-k candidates
        k = config.k_list[step]
        InitialArray,BestMape = SelectTopK(InitialArray, BestMape, step, k)

        F_l = config.F_list[step][0]
        F_u = config.F_list[step][1]
        MutatedArray = DEAlgorithmClass.mutation(InitialArray,F_l,F_u)
        CrossArray = DEAlgorithmClass.crossover(InitialArray, MutatedArray)

        if step == 0:
            args = InitialArray + CrossArray
            CombineArray = args
        else:
            args = CrossArray
            CombineArray = InitialArray + CrossArray

        mapelist = []
        for num in range(len(args)):
            item = args[num]
            try:
                res,fir_epoch_loss = Train(item, round_no, Global_fir_epoch_loss, f, num, step)
            except:
                res, fir_epoch_loss = 100.0, 10 ** 10

            mapelist.append(res)

            if num==0 and step==0:
                Global_fir_epoch_loss = fir_epoch_loss
                Global_step_mape = res
            else:
                if res<Global_step_mape:
                    Global_fir_epoch_loss = fir_epoch_loss
                    Global_step_mape = res

        if step == 0:
            mapelist = mapelist
        else:
            mapelist = BestMape + mapelist

        StepMinVal = min(mapelist)

        if StepMinVal < GlobalMinVal:
            LIdx = mapelist.index(StepMinVal)
            StepOptimalPara = CombineArray[LIdx]
            GlobalOptimalPara = StepOptimalPara
            GlobalMinVal = StepMinVal

        printtxt = "Step: %s, min MAPE: %s, hyperparameter: %s" % (step, GlobalMinVal, GlobalOptimalPara)
        print(printtxt)
        os.write(f, str.encode(printtxt + '\n'))

        assert len(InitialArray) == len(CrossArray) == k
        SelectArray, BestMape = DEAlgorithmClass.selection(mapelist, InitialArray, CrossArray, k)
        InitialArray = SelectArray

        # early stopping applied
        GlobalMinValLi.append(GlobalMinVal)
        if step + 1 >= config.EarlyStopStep:
            if GlobalMinValLi[-1] == GlobalMinValLi[-2] == GlobalMinValLi[-3]:
                break

    duration = (datetime.now() - start).total_seconds()/3600
    printtxt = "computational time: %s h" %(duration)
    print(printtxt)

    # update best para table
    Df_Para = pd.read_csv(config.ParaSavePath)
    Df_step = pd.DataFrame({'round_no': [round_no], 'best_para': [GlobalOptimalPara], 'time': [duration], 'best_fitness': [GlobalMinVal]})
    Df_Para = pd.concat([Df_Para, Df_step], axis=0)
    Df_Para.to_csv(config.ParaSavePath, index=False)

    return None


if __name__ == "__main__":
    global x_train_glo, y_train_glo, x_val_glo, y_val_glo
    x_train_glo, y_train_glo, x_val_glo, y_val_glo = dataset(50)

    RunDE(0)
