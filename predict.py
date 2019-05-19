#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

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
        self._data = np.log(df.dropna(how='any')[df.Volume>0.])

    def _save(self):
        with open('config.json','w') as f:
            f.write(self._model.to_json())
        self._model.save_weights('N225.hdf5')

    def _load(self):
        with open('config.json','r') as f:
            self._model = model_from_json(f.read())
        self._model.load_weights('N225.hdf5')

    def _series(self,data):
        d=[]
        for day in range(self._config['term']):
            d.append(data[day:].reset_index(drop=True))
        f_data = pd.concat(d,axis=1).dropna(how='any')

        #print( np.log(f_data[0:1]).values[0] )
        return f_data[0:1].values[0]

    def _trade(self,data):
        f_data = data.reset_index(drop=True)

        buy = f_data.at[0,'Open']
        last = self._config['keep']-1
        threshold = self._config['change']+1.

        category = np.zeros(last+1)
        for day in range(1,last):
            sell = f_data.at[day,'Open']
            keep = f_data.at[day+1,'Open']
            if buy + np.log(threshold) < sell and sell > keep:
                category[day]=1
                break
        else:
            sell = f_data.at[last,'Open']
            if buy + np.log(threshold) < sell:
                category[last]=1
            else:
                category[0]=1
        assert sum(category)==1, '条件漏れ'

        return category

    def _mkdataset(self):
        term = self._config['term']
        keep = self._config['keep']
        k_end = len(self._data)

        dataset = []
        label   = []
        recent  = []
        for k in range(term,k_end-keep):
            data = self._series(self._data[k-term:k])
            dataset.append(data)
        for k in range(term,k_end-keep):
            data = self._trade(self._data[k:k+keep])
            label.append(data)
        for k in range(k_end-keep,k_end):
            data = self._series(self._data[k-term:k])
            recent.append(data)

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        self._x = np.reshape(scaler.fit_transform(dataset),
            (len(dataset),term,self._data.shape[1]))
        self._y = np.reshape(label,(len(dataset),keep))
        self._z = np.reshape(scaler.fit_transform(recent),
            (len(recent),term,self._data.shape[1]))

    def _model(self):
        days = self._config['term']
        dimension = 6
        model = Sequential()
        model.add(LSTM(
            self._ml['hidden'],
            use_bias=True,
            dropout=0.5,
            recurrent_dropout=0.5,
            return_sequences=False,
            batch_input_shape=(None, days, dimension)))
        model.add(Dense(self._config['keep']))
        model.add(Activation("linear"))
        optimizer = Adam(lr=0.001)
        model.compile(loss="mean_squared_error", optimizer=optimizer)
        model.fit(self._x, self._y, epochs=100, batch_size=512, validation_split=0.1)
        self._model = model

    def _predict(self):
        ans = self._model.predict(self._z)
        print(np.round(ans,decimals=2))


if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    #parser.add_argument('-l','--load_model',action='store_true')
    parser.add_argument('-c','--calculate_model',action='store_true')
    args = parser.parse_args()

    a=Kabu()
    a._read()
    a._mkdataset()
    if(args.calculate_model):
        a._model()
        a._predict()
        a._save()
    else:
        a._load()
        a._predict()
