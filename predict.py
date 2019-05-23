#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import scipy as sp
import scipy.fftpack

from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Activation, Dropout, InputLayer, Bidirectional, Input, Multiply
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

class Kabu:
    def __init__(self,filename='^N225.csv'):
        self._data =[]
        self._filename = filename
        self._config = {
            'keep':3,
            'term':64,
            'category':(-.07,-.03,-.01,-.005,.0,+.005,+.01,+.03,+.07),
            }
        #self._config = {'keep':2,'term':25,'change':0.03,'cat':(-.03,-.01,.0,+.01,+.03)}
        self._ml = {'hidden':500,'epoch':100,'batch':64}
        self._x = []
        self._y = []
        self._z = []

    def _read(self):
        self._data = pd.read_csv(self._filename,index_col=0)
        #self._data = self._data[self._data.Volume>0.]
        self._data = self._data.drop('Volume',axis=1)
        self._data = self._data.dropna(how='any')
        self._data = np.log(self._data)[-500:]

    def _save(self):
        with open(self._filename+'.json','w') as f:
            f.write(self._model.to_json())
        self._model.save_weights(self._filename+'.hdf5')

    def _load(self):
        with open(self._filename+'.json','r') as f:
            self._model = model_from_json(f.read())
        self._model.load_weights(self._filename+'.hdf5')

    def _rule(self,data):

        output = []
        for k in data.index:
            #翌日購入,翌々日売却
            buy = data.at[k,(1,'Open')]
            sell = data.at[k,(1,'Close')]
            category = np.zeros(len(self._config['category'])+1)

            for j,theta in enumerate(self._config['category']):
                if sell - buy < np.log(1+theta):
                    category[j] = 1.
                    break
            else:
                category[len(self._config['category'])] = 1.
            assert sum(category)==1, '条件漏れ'
            output.append(category)

        output = pd.DataFrame(output,index=data.index)

        return output

    def _mkDataset(self):
        term = self._config['term']
        keep = self._config['keep']
        scaler = MinMaxScaler(feature_range=(0, 1))

        #当日を含めてterm日間のデータを横に並べる
        before = pd.concat([self._data.shift(+k) for k in range(term)], axis=1, keys=range(term))
        before = before.dropna(how='any')

        #翌日からkeep日間のデータを横に並べる
        after = pd.concat([self._data.shift(-k) for k in range(keep)], axis=1, keys=range(keep))
        after = after.dropna(how='any')
        after = after[after.index.isin(before.index)]

        #無駄な処理だが、Pandasを維持するため、NumPyにする直前でMinMax
        #1次元にするとMinMaxできないので、二次元化する
        dataset = np.reshape(
            scaler.fit_transform(before.values.flatten().reshape(-1,1)),
            [len(before.index), self._config['term'], len(self._data.columns)])
        label = self._rule(after)
        dataset2 = np.reshape(
            scaler.fit_transform(before.sort_index(axis=1).values.flatten().reshape(-1,1)),
            [len(before.index), len(self._data.columns), self._config['term']])
        #離散コサイン変換
        wave = sp.fftpack.dct(dataset2,axis=2)

        self._y = label.values
        self._x,self._z = np.split(dataset,[len(self._y)])
        self._wx,self._wz = np.split(wave,[len(self._y)])

    def _mkModel2(self):
        days = self._config['term']
        dimension = len(self._data.columns)

        input_raw = Input(shape=(days,dimension))
        lstm_1 = LSTM(
            self._ml['hidden'],
            return_sequences=False,
            input_shape=(days, dimension),
            activation='relu')(input_raw)
        drop_1 = Dropout(.1)(lstm_1)

        input_wav = Input(shape=(dimension,days))
        lstm_2 = LSTM(
            self._ml['hidden'],
            return_sequences=False,
            input_shape=(dimension, days),
            activation='relu')(input_wav)
        drop_2 = Dropout(.2)(lstm_2)

        multiplied = Multiply()([drop_1,drop_2])
        dense_1 = Dense(75)(multiplied)
        dense_2 = Dense(75)(dense_1)
        dense_3 = Dense(
            len(self._y[0]),
            kernel_initializer='glorot_uniform')(dense_2)
        output = Activation('softmax')(dense_3)

        model = Model(inputs=[input_raw,input_wav],outputs=output)
        optimizer = Adam(lr=0.001)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.fit([self._x,self._wx], self._y, epochs=self._ml['epoch'], batch_size=self._ml['batch'], validation_split=0.2)
        self._model = model
        pass

    def _mkModel(self):
        days = self._config['term']
        dimension = len(self._data.columns)
        model = Sequential()
        #model.add(InputLayer(input_shape=(days,dimension)))
        model.add(LSTM(
            self._ml['hidden'],
            return_sequences=False,
            input_shape=(days, dimension),
            activation='relu'))
            #batch_input_shape=(None, days, dimension)))
        model.add(Dropout(0.2))
        model.add(Dense(75,activation='relu'))
        model.add(Dense(75,activation='relu'))
        model.add(Dense(len(self._y[0]),kernel_initializer='glorot_uniform'))
        model.add(Activation('softmax'))
        optimizer = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.fit(self._x, self._y, epochs=self._ml['epoch'], batch_size=self._ml['batch'], validation_split=0.2)
        self._model = model

    def _predict(self):
        ans = self._model.predict([self._z,self._wz])
        print(np.round(ans,decimals=2))

    def _predict2(self):
        ans = self._model.predict([self._x,self._wx])
        ans = list(zip(self._y,ans))
        for input,output in np.round(ans,decimals=2):
            print(input,output)

if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-c','--calculate_model',action='store_true')
    parser.add_argument('-v','--visualize',action='store_true')
    parser.add_argument('-f','--filename',type=str,default='^N225.csv')
    parser.add_argument('-a','--compare_all',action='store_true')
    args = parser.parse_args()

    a=Kabu(filename=args.filename)
    a._read()
    if(args.visualize):
        from keras.utils import plot_model
        a._load()
        a._model.summary()
    elif(args.calculate_model):
        a._mkDataset()
        a._mkModel2()
        a._predict()
        a._save()
    elif(args.compare_all):
        a._load()
        a._mkDataset()
        a._predict2()
    else:
        a._load()
        a._mkDataset()
        a._predict()
