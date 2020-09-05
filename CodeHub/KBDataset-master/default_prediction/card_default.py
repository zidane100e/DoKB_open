# class for card default open data
# from http://archive.ics.uci.edu/ml
 
import sys, os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

from kutils import data as data_util

class Data():
    x_train, y_train = None, None
    x_test, y_test = None, None

    def __init__(self):
        self.f_in_s = 'UCI_Credit_Card.csv'
        self.f_out_s = 'default_prediction_UCI.npz'        

    def load_data(self):
        if os.path.exists(self.f_out_s):
            data0 = np.load(self.f_out_s)
            Data.x_train, Data.y_train = data0['xs_train'], data0['ys_train']
            Data.x_test, Data.y_test = data0['xs_test'], data0['ys_test']
        else:
            self.preprocess()
        return (Data.x_train, Data.y_train), (Data.x_test, Data.y_test)

    def preprocess(self):
        xs, ys = self.read_txt(self.f_in_s)

        # normalize amount data
        xs = self.normalize(xs)

        # separate data into training set and test set with ratio
        n_data = xs.shape[0]
        ids = list(range(n_data))
        np.random.shuffle(ids)
        ratio = int(0.75*n_data)

        xs_train = xs.iloc[ids[:ratio]]
        ys_train = ys.iloc[ids[:ratio]]
        xs_test = xs.iloc[ids[ratio:]]
        ys_test = ys.iloc[ids[ratio:]]

        # correct some errors in the original data and 
        # represent category variables into one-hot encoding
        self.df_temp = xs_train
        xs_train = self.arrange().modify_hotcode().df_temp
        self.df_temp = xs_test
        xs_test = self.arrange().modify_hotcode().df_temp
        
        Data.x_train = xs_train.values
        Data.y_train = ys_train.values
        Data.x_test = xs_test.values
        Data.y_test = ys_test.values

        np.savez(self.f_out_s, xs_train=Data.x_train, ys_train=Data.y_train, xs_test=Data.x_test, ys_test=Data.y_test)

    def read_txt(self, f1_s, delimiter = ','):
        map_strip = lambda x: x.strip()
        map_int = lambda x: int(float(x))
        xs, ys = [], []
        with open(f1_s) as f1:
            columns_s = f1.readline().strip().split(delimiter)
            columns_s = list(map(lambda x: x.strip('"'), columns_s))
            columns_sx = columns_s[1:-1]
            columns_sy = [columns_s[-1]]
            for line in f1:
                elms = list(map(map_strip, line.strip().split(delimiter)))
                xs.append(list(map(map_int, elms[1:-1])))
                ys.append(int(float(elms[-1])))
        df1 = pd.DataFrame(data=xs, columns = columns_sx)
        df2 = pd.DataFrame(data=ys, columns = columns_sy)
        return df1, df2

    def normalize(self, df1):
        vars0 = ['BILL_AMT', 'PAY_AMT', 'PAY_']
        vars = [vars0[0]+str(x) for x in range(1, 6+1)]
        vars += [vars0[1] + str(x) for x in range(1, 6 + 1)]
        vars += [vars0[2] + str(x) for x in range(2, 6 + 1)] # not from 1
        vars += ['PAY_0']
        vars += ['LIMIT_BAL', 'AGE']
        for var1 in vars:
            df1[var1] = minmax_scale(df1[var1])
        return df1

    # arrange some categories not in the explanation
    def arrange(self):
        # make marriage status 0 to 3
        self.df_temp.loc[self.df_temp.MARRIAGE==0, 'MARRIAGE'] = 3
        # make education status 0, 6
        self.df_temp.loc[(self.df_temp.EDUCATION == 0) | (self.df_temp.EDUCATION == 6), 'EDUCATION'] = 5
        return self

    # apply one hot encoding
    def modify_hotcode(self):
        strs = ['SEX', 'MARRIAGE', 'EDUCATION']
        for str1 in strs:
            self.modify_hotcode_column(str1)
        return self

    # one hot encoding for a column
    def modify_hotcode_column(self, col_s):
        hotcodes = data_util.onehotencode(self.df_temp[col_s])
        hotcodest = data_util.transpose(hotcodes)
        for i in range(len(hotcodes[0])):
            col_s1 = col_s + '_' + str(i)
            self.df_temp[col_s1] = hotcodest[i]
        self.df_temp = self.df_temp.drop(col_s, axis=1)

if __name__ == '__main__':
    
    data1 = Data()
    data1.load_data()
    (x_train, y_train), (x_test, y_test) = data1.load_data()
    print(x_train, y_train, x_test, y_test)
