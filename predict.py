#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

class Kabu:
    def __init__(self,filename='^N225.csv'):
        self._data =[]
        self._filename = filename
        self._config = {'keep':10,'term':25,'change':0.03}
        self._ml = {'hidden':300,}
        self._x = []
        self._y = []

    def _read(self):
        df = pd.read_csv(self._filename,index_col=0)
        self._data = np.log(df.dropna(how='any')[df.Volume>0.])[-2000:]

    def _save(self):
        with open(self._filename+'.json','w') as f:
            f.write(self._model.to_json())
        self._model.save_weights(self._filename+'.hdf5')

    def _load(self):
        with open(self._filename+'.json','r') as f:
            self._model = model_from_json(f.read())
        self._model.load_weights(self._filename+'.hdf5')

    def _series(self,data):
        d=[]
        for day in range(self._config['term']):
            d.append(data[day:].reset_index(drop=True))
        f_data = pd.concat(d,axis=1).dropna(how='any')

        #print( np.log(f_data[0:1]).values[0] )
        return f_data[0:1].values[0]

    def _trade(self,data):
        f_data = data.reset_index(drop=True)

        #翌日購入
        buy = f_data.at[1,'Open']
        #最大last日後まで保有
        last = self._config['keep']+1
        #最低利益
        threshold = self._config['change']+1.

        category = np.zeros(last-1)
        #最終日前日でなく、翌日値上がり期待するなら売らない
        #最終日前日でなく、翌日値上がり期待できないなら売る
        for day in range(2,last-1):
            sell = f_data.at[day,'Open']
            keep = f_data.at[1+day,'Open']
            if buy + np.log(threshold) < sell and sell > keep:
                category[day-1]=1
                break
        #最終日前日なら売る
        else:
            sell = f_data.at[last-1,'Open']
            if buy + np.log(threshold) < sell:
                category[last-2]=1
            #買うべきでない
            else:
                category[0]=1
        assert sum(category)==1, '条件漏れ'

        return category

    def _mkDataset(self):
        term = self._config['term']
        keep = self._config['keep']
        k_end = len(self._data)
        scaler = MinMaxScaler(feature_range=(0, 1))

        dataset = []
        label   = []
        recent  = []
        for k in range(term,k_end):
            data = self._series(self._data[k-term:k])
            dataset.append(data)
        for k in range(term,k_end-keep-1):
            data = self._trade(self._data[k:k+keep+1])
            label.append(data)
        '''
        for k in range(k_end-keep-1,k_end):
            data = self._series(self._data[k-term:k])
            recent.append(data)
        '''

        x = np.reshape(scaler.fit_transform(dataset),
            (len(dataset),term,self._data.shape[1]))
        self._x,self._z = np.split(x,[len(label)])
        self._y = np.reshape(label,(len(label),keep))

    def _mkModel(self):
        days = self._config['term']
        dimension = 6
        model = Sequential()
        model.add(LSTM(
            self._ml['hidden'],
            use_bias=True,
            dropout=0.5,
            recurrent_dropout=0.5,
            return_sequences=False,
            activation='relu',
            batch_input_shape=(None, days, dimension)))
        model.add(Dense(75))
        model.add(Dense(25))
        model.add(Dense(self._config['keep'],activation='softmax'))
        optimizer = Adam(lr=0.001)
        model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['accuracy'])
        model.fit(self._x, self._y, epochs=100, batch_size=512, validation_split=0.1)
        self._model = model

    def _predict(self):
        ans = self._model.predict(self._z)
        print(np.round(ans,decimals=2))


if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-c','--calculate_model',action='store_true')
    parser.add_argument('-p','--plot',action='store_true')
    args = parser.parse_args()

    a=Kabu()
    a._read()
    if(args.plot):
        from keras.utils import plot_model
        a._load()
        plot_model(a._model, to_file="model.png", show_shapes=True)
    elif(args.calculate_model):
        a._mkDataset()
        a._mkModel()
        a._predict()
        a._save()
    else:
        a._load()
        a._mkDataset()
        a._predict()
