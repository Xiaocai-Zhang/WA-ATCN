# -*- coding: utf-8 -*-
import pandas as pd
pd.options.mode.chained_assignment = None
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from I280.WFE import bulid_wfe
import numpy as np


class processing:
    def __init__(self, df, time_step, date_start, date_split, train=True):
        self.df = df
        self.time_step = time_step
        self.date_start = date_start
        self.date_split = date_split
        self.train = train


    def proc_data(self,df):
        cols = ['hr_emb_1','hr_emb_2','wk_emb_1','wk_emb_2']
        df[cols] = df.apply(lambda row: bulid_wfe(row['timestamp']).cfeVec(),axis=1,result_type='expand')
        x_train_li = []
        y_train_li = []
        x_test_li = []
        y_test_li = []
        if self.train:
            for index,_ in df.iterrows():
                if index<len(df)-self.time_step:
                    df_f = df.loc[index:index+self.time_step-1,['flow']+cols]
                    df_f_array = df_f.to_numpy()
                    label = df.loc[index+self.time_step, 'flow']
                    time = df.loc[index+self.time_step, 'timestamp']
                    if time<self.date_split:
                        if time >= self.date_start:
                            x_train_li.append(df_f_array)
                            y_train_li.append(label)
                    else:
                        break

            x_train = np.array(x_train_li, np.float32)
            y_train = np.array(y_train_li, np.float32)
            y_train = np.reshape(y_train, newshape=(-1, 1))

            return x_train, y_train
        else:
            for index,_ in df.iterrows():
                if index<len(df)-self.time_step:
                    df_f = df.loc[index:index+self.time_step-1,['flow']+cols]
                    df_f_array = df_f.to_numpy()
                    label = df.loc[index+self.time_step, 'flow']
                    time = df.loc[index+self.time_step, 'timestamp']
                    if time<self.date_split:
                        if time >= self.date_start:
                            x_train_li.append(df_f_array)
                            y_train_li.append(label)
                    else:
                        x_test_li.append(df_f_array)
                        y_test_li.append(label)

            x_train = np.array(x_train_li, np.float32)
            y_train = np.array(y_train_li, np.float32)
            y_train = np.reshape(y_train,newshape=(-1,1))
            x_test = np.array(x_test_li, np.float32)
            y_test = np.array(y_test_li, np.float32)
            y_test = np.reshape(y_test, newshape=(-1, 1))

            return x_train, y_train, x_test, y_test


    def main(self):
        if self.train:
            x_train, y_train = self.proc_data(self.df)
            return x_train, y_train
        else:
            x_train, y_train, x_test, y_test = self.proc_data(self.df)
            return x_train, y_train, x_test, y_test