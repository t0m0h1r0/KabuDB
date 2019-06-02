#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import scipy as sp
import scipy.fftpack
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

from keras.models import Sequential, model_from_json, load_model, Model
from keras.layers import Dense, Activation, Dropout, InputLayer, Bidirectional, Input, Multiply, Concatenate
from keras.layers.recurrent import LSTM, RNN, SimpleRNN, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.utils.training_utils import multi_gpu_model

from sklearn.preprocessing import MinMaxScaler

class Kabu:
    def __init__(self,filename='^N225.csv'):
        self._data =[]
        self._filename = filename
        self._config = {
            'days':4000,
            'keep':3,
            'term':256,
            #'category':(-.3,.0,+.3)
            'category':(-.07,-.03,-.01,-.005,.0,+.005,+.01,+.03,+.07),
            }
        self._ml = {'hidden':200,'epoch':5000,'batch':64}
        self._x = []
        self._y = []
        self._z = []
        self._wx = []
        self._wz = []

    def _read(self):
        self._data = pd.read_csv(self._filename,index_col=0)
        self._data = self._data.drop('Volume',axis=1)
        '''
        self._data = self._data.drop('Close',axis=1)
        self._data = self._data.drop('High',axis=1)
        self._data = self._data.drop('Low',axis=1)
        self._data = self._data.drop('Adj Close',axis=1)
        '''
        self._data = self._data.dropna(how='any')
        self._data = self._data[-self._config['days']:]
        #self._data = np.log(self._data)[-self._config['days']:]

    def _save(self):
        with open(self._filename+'.json','w') as f:
            f.write(self._model_for_save.to_json())
        self._model.save_weights(self._filename+'.hdf5')
        self._model_for_save.save(self._filename+'.h5')

    def _load(self):
        self._model = load_model(self._filename+'.h5')
        '''
        with open(self._filename+'.json','r') as f:
            self._model = model_from_json(f.read())
        self._model.load_weights(self._filename+'.hdf5')
        '''

    def _rule1(self,data):
        diff = []
        borders = np.concatenate(
            [[-float('inf')],
            self._config['category'],
            [float('inf')]])
        for k in data.index:
            #翌日購入,翌々日売却
            buy = data.at[k,(1,'Open')]
            sell = data.at[k,(2,'Open')]
            diff.append(sell/buy-1.)
            #diff.append(sell-buy)
        nums, bins = pd.cut(diff, borders, labels=range(len(borders)-1),retbins=True)
        output = to_categorical(nums)
        output = pd.DataFrame(output,index=data.index)
        print((bins)*100.)
        #print((np.exp(bins)-1.)*100.)
        return output

    def _rule2(self,data,counts=6):
        diff = []
        for k in data.index:
            #翌日購入,翌々日売却
            buy = data.at[k,(1,'Open')]
            sell = data.at[k,(2,'Open')]
            diff.append(sell/buy-1.)
            #diff.append(sell-buy)
        nums, bins = pd.qcut(diff,counts,labels=range(counts),retbins=True)
        output = to_categorical(nums)
        output = pd.DataFrame(output,index=data.index)
        print((bins)*100.)
        #print((np.exp(bins)-1.)*100.)
        return output

    def _rule3(self,data):
        output = data.loc[:,(1,slice(None))].sort_index(axis=1,level=(1,0))
        return output

    def _generate(self):
        term = self._config['term']
        keep = self._config['keep']
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        #data = self._data
        data = pd.DataFrame(self._scaler.fit_transform(self._data.values),index=self._data.index)

        #当日を含めてterm日間のデータを横に並べる
        before = pd.concat([data.shift(+k) for k in range(term)], axis=1, keys=range(term))
        before = before.dropna(how='any')
        before = before.sort_index(axis=1, level=(0,1))

        #当日からkeep日間のデータを横に並べる
        after = pd.concat([data.shift(-k) for k in range(keep)], axis=1, keys=range(keep))
        #after = pd.concat([self._data.shift(-k) for k in range(keep)], axis=1, keys=range(keep))
        after = after.dropna(how='any')
        after = after.sort_index(axis=1, level=(0,1))
        after = after[after.index.isin(before.index)]

        #無駄な処理だが、Pandasを維持するため、NumPyにする直前でMinMax
        #1次元にするとMinMaxできないので、二次元化する
        dataset = np.reshape(
            before.values.flatten().reshape(-1,1),
            [len(before.index), self._config['term'], len(data.columns)])
            #[len(before.index), len(self._data.columns), self._config['term']])
        label = self._rule3(after)
        dataset2 = np.reshape(
            before.sort_index(axis=1,level=1).values.flatten().reshape(-1,1),
            [len(before.index), len(data.columns), self._config['term']])
        #離散フーリエ変換
        wave = np.abs(sp.fftpack.fft(dataset2,axis=2))
        #print(wave)

        self._y = label.values
        self._x,self._z = np.split(dataset,[len(self._y)])
        self._wx,self._wz = np.split(wave,[len(self._y)])

    def _build(self,gpus=1):
        days = self._config['term']
        dimension = len(self._data.columns)

        input_raw = Input(shape=(days,dimension))
        drop_a1 = Dropout(.2)(input_raw)
        lstm_a1 = Bidirectional(GRU(
            self._ml['hidden'],
            #use_bias=True,
            #return_sequences=False,
            return_sequences=True,
            input_shape=(dimension, days),
            activation='tanh'))(drop_a1)
        drop_a1 = Dropout(.95)(lstm_a1)
        lstm_a1 = Bidirectional(GRU(
            self._ml['hidden'],
            activation='relu'))(drop_a1)
        drop_a1 = Dropout(.95)(lstm_a1)
        '''
        input_wav = Input(shape=(dimension,days))
        drop_b1 = Dropout(.2)(input_wav)
        lstm_b1 = Bidirectional(GRU(
        #lstm_b = Bidirectional(LSTM(
            self._ml['hidden'],
            use_bias=True,
            return_sequences=False,
            #return_sequences=True,
            input_shape=(dimension, days),
            activation='relu'))(drop_b1)
        drop_b1 = Dropout(.5)(lstm_b1)

        merged = Concatenate()([drop_a1,drop_b1])
        '''
        merged = drop_a1
        dense_2 = Dense(
            len(self._y[0]),
            kernel_initializer='he_normal')(merged)
        output = Activation('relu')(dense_2)
        #output = Activation('softmax')(dense_2)

        #model = Model(inputs=input_raw,outputs=output)
        model = Model(inputs=[input_raw],outputs=output)
        #model = Model(inputs=[input_raw,input_wav],outputs=output)
        optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)

        self._model_for_save = model
        if gpus>1:
            model = multi_gpu_model(model,gpus=gpus)
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        #model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self._model = model

    def _calculate(self):
        early_stopping = EarlyStopping(patience=10, verbose=1)
        self._model.fit(
            [self._x], self._y,
            #[self._x,self._wx], self._y,
            epochs=self._ml['epoch'],
            batch_size=self._ml['batch'],
            validation_split=0.2,
            shuffle=False,
            callbacks=[early_stopping])

    def _predict(self):
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        ans = self._model.predict([self._z])
        ans = self._scaler.inverse_transform(ans)
        #ans = self._model.predict([self._z,self._wz])
        print(np.round(ans,decimals=2))

    def _validate(self):
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        ans = self._model.predict([self._x])
        ans = self._scaler.inverse_transform(ans)
        cal = self._scaler.inverse_transform(self._y)
        #ans = self._model.predict([self._x,self._wx])
        ans = list(zip(cal,ans))
        for input,output in np.round(ans,decimals=2):
            print(input,output,'=>',np.dot(input,output))

if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-l','--learn',action='store_true')
    parser.add_argument('-v','--visualize',action='store_true')
    parser.add_argument('-f','--csv_filename',type=str,default='^N225.csv')
    parser.add_argument('-a','--compare_all',action='store_true')
    parser.add_argument('-g','--gpus',type=int,default=1)
    args = parser.parse_args()

    a=Kabu(filename=args.csv_filename)
    a._read()
    if(args.visualize):
        from keras.utils import plot_model
        a._load()
        a._model.summary()
        plot_model(a._model, to_file='model.png')
    elif(args.learn):
        a._generate()
        a._build(gpus=args.gpus)
        a._model.summary()
        a._calculate()
        a._predict()
        a._save()
    elif(args.compare_all):
        a._load()
        a._generate()
        a._validate()
    else:
        a._load()
        a._generate()
        a._predict()
