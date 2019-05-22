#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout, InputLayer, Bidirectional
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
            'term':25,
            'change':0.03,
            'category':(-.03,-.01,.0,+.01,+.03),
            }
        #self._config = {'keep':2,'term':25,'change':0.03,'cat':(-.03,-.01,.0,+.01,+.03)}
        self._ml = {'hidden':500,'epoch':50,'batch':32}
        self._x = []
        self._y = []
        self._z = []

    def _read(self):
        self._data = pd.read_csv(self._filename,index_col=0)
        #self._data = self._data[self._data.Volume>0.]
        self._data = self._data.drop('Volume',axis=1)
        #self._data = self._data.drop('Adj Close',axis=1)
        #self._data = self._data.drop('Close',axis=1)
        #self._data = self._data.drop('High',axis=1)
        #self._data = self._data.drop('Low',axis=1)
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
            sell = data.at[k,(2,'Open')]
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
        k_end = len(self._data)
        scaler = MinMaxScaler(feature_range=(0, 1))

        label   = []
        recent  = []
        before = pd.concat([self._data.shift(+k) for k in range(term)], axis=1, keys=range(term))
        before = before.dropna(how='any')

        after = pd.concat([self._data.shift(-k) for k in range(keep)], axis=1, keys=range(keep))
        after = after.dropna(how='any')
        after = after[after.index.isin(before.index)]

        dataset = np.reshape(before.values,
            [len(before.index), self._config['term'], len(self._data.columns)])
        label = self._rule(after)

        self._y = label.values
        self._x,self._z = np.split(dataset,[len(self._y)])

    def _mkModel(self):
        days = self._config['term']
        dimension = len(self._data.columns)
        model = Sequential()
        model.add(InputLayer(input_shape=(days,dimension)))
        model.add(LSTM(
            self._ml['hidden'],
            return_sequences=False,
            #input_shape=(days, dimension),
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
        ans = self._model.predict(self._z)
        print(np.round(ans,decimals=2))

    def _predict2(self):
        ans = self._model.predict(self._x)
        ans = list(zip(self._y,ans))
        for input,output in np.round(ans,decimals=2):
            print(input,output)

if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-c','--calculate_model',action='store_true')
    parser.add_argument('-p','--plot',action='store_true')
    parser.add_argument('-f','--filename',type=str,default='^N225.csv')
    parser.add_argument('-a','--compare_all',action='store_true')
    args = parser.parse_args()

    a=Kabu(filename=args.filename)
    a._read()
    if(args.plot):
        from keras.utils import plot_model
        a._load()
        a._model.summary()
    elif(args.calculate_model):
        a._mkDataset()
        a._mkModel()
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
