#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import scipy as sp
import scipy.fftpack
import scipy.signal
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

from keras.models import Sequential, model_from_json, load_model, Model
from keras.layers import Dense, Activation, Dropout, InputLayer, Bidirectional, Input, Multiply, Concatenate, SpatialDropout1D
from keras.layers.recurrent import LSTM, RNN, SimpleRNN, GRU
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.utils import multi_gpu_model

from sklearn.preprocessing import MinMaxScaler, PowerTransformer, FunctionTransformer

from qrnn import QRNN

class KabuQRNN:
    def __init__(self,filename='^N225.csv'):
        self._data =[]
        self._filename = filename
        self._config = {
            'days':1000,
            'keep':2,
            'term':128,
            #'category':(-.3,.0,+.3)
            'category':(-.07,-.03,-.01,-.005,.0,+.005,+.01,+.03,+.07),
            }
        self._ml = {'hidden':512,'epoch':5000,'batch':1024}
        self._scaler = MinMaxScaler(feature_range=(-1, 1))
        #self._scaler = PowerTransformer()
        #self._scaler = FunctionTransformer(func=lambda x:x, inverse_func=lambda x:x)
        self._x = []
        self._y = []
        self._z = []
        self._wx = []
        self._wz = []

    def _read(self):
        self._data = pd.read_csv(self._filename,index_col=0)
        self._data = self._data.drop('Volume',axis=1)
        self._data = self._data.drop('Adj Close',axis=1)
        '''
        self._data = self._data.drop('Close',axis=1)
        self._data = self._data.drop('High',axis=1)
        self._data = self._data.drop('Low',axis=1)
        self._data = self._data.drop('Adj Close',axis=1)
        '''
        self._data = self._data.dropna(how='any')
        self._data = self._data[-self._config['days']:]
        self._data = self._data.sort_index(axis=1)
        #self._data = np.log(self._data)[-self._config['days']:]

    def _save(self,model):
        model.save(self._filename+'.h5')

    def _load(self,model):
        #model = load_model(self._filename+'.h5')
        model.load_weights(self._filename+'.h5')

    def _generate(self):
        term = self._config['term']
        keep = self._config['keep']
        data = pd.DataFrame(self._scaler.fit_transform(self._data.values),
            index=self._data.index, columns=self._data.columns)

        #当日を含めてterm日間のデータを横に並べる
        before = pd.concat([data.shift(+k) for k in range(term)], axis=1, keys=range(term))
        before = before.dropna(how='any')
        before = before.sort_index(axis=1, level=(0,1))

        #翌日のデータ
        after = data.shift(-1)
        after = after.dropna(how='any')
        after = after.sort_index(axis=1, level=(0,1))
        after = after[after.index.isin(before.index)]

        #入力データ1
        dataset = np.reshape(
            before.values.flatten().reshape(-1,1),
            [len(before.index), self._config['term'], len(data.columns)])

        #入力データ2
        dataset2 = np.reshape(
            before.sort_index(axis=1,level=(1,0)).values.flatten().reshape(-1,1),
            [len(before.index), len(data.columns), self._config['term']])
        #離散フーリエ変換
        #wave = np.abs(sp.fftpack.fft(wave,axis=2))
        #wave = np.abs(sp.fftpack.dct(wave,axis=2))
        #wave = sp.fftpack.dct(dataset2,axis=2)
        #wave = wave / float(wave.shape[2])

        import pywt
        wave = pywt.wavedec(dataset2, wavelet='haar', axis=2)
        wave = np.concatenate(wave,axis=2)


        y = after.values
        x,z = np.split(dataset,[len(y)])
        wx,wz = np.split(wave,[len(y)])

        return [x,wx],y,[z,wz]

    def _build(self,gpus=1):
        days = self._config['term']
        dimension = len(self._data.columns)
        window=days

        input_raw = Input(shape=(days,dimension))
        x = input_raw
        x = SpatialDropout1D(0.2)(x)
        x = QRNN(
            units= self._ml['hidden'],
            window_size=window,
            return_sequences=False,
            stride=1,
            )(x)

        input_wav = Input(shape=(dimension,days))
        y = input_wav
        y = SpatialDropout1D(0.2)(y)
        y = QRNN(
            units= self._ml['hidden'],
            window_size=window,
            return_sequences=False,
            stride=1,
            )(y)

        merged = Concatenate()([x,y])
        label = Dense( units= dimension )(merged)
        output = Activation('sigmoid')(label)

        model = Model(inputs=[input_raw,input_wav],outputs=output)
        optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)

        base = model
        if gpus>1:
            model = multi_gpu_model(model,gpus=gpus)
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return model,base

    def _calculate(self,model,x,y):
        early_stopping = EarlyStopping(patience=50, verbose=1)
        model.fit(
            x, y,
            epochs=self._ml['epoch'],
            batch_size=self._ml['batch'],
            validation_split=0.2,
            shuffle=False,
            callbacks=[early_stopping])

    def _predict(self,model,z):
        ans = model.predict(z)
        ans = self._scaler.inverse_transform(ans)
        #ans = self._model.predict([self._z,self._wz])
        print(np.round(ans,decimals=2))
        return ans

    def _validate(self,model,x,y):
        ans = model.predict(x)
        ans = self._scaler.inverse_transform(ans)
        cal = self._scaler.inverse_transform(y)
        cal[0] = np.multiply(cal[0],float('nan'))
        cal=np.roll(cal,-1, axis=0)
        #ans = self._model.predict([self._x,self._wx])
        ans = list(zip(cal,ans))
        for input,output in np.round(ans,decimals=2):
            print(input,output,'=>',input-output)

class KabuLSTM(KabuQRNN):
    def _build(self,gpus=1):
        days = self._config['term']
        dimension = len(self._data.columns)
        window=days

        input_raw = Input(shape=(days,dimension))
        x = input_raw
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(
            units= self._ml['hidden'],
            return_sequences=False,
            ))(x)

        input_wav = Input(shape=(dimension,days))
        y = input_wav
        y = Dropout(0.2)(y)
        y = Bidirectional(LSTM(
            units= self._ml['hidden'],
            return_sequences=False,
            ))(y)

        merged = Concatenate()([x,y])
        label = Dense( units= dimension )(merged)
        output = Activation('sigmoid')(label)

        model = Model(inputs=[input_raw,input_wav],outputs=output)
        optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)

        base = model
        if gpus>1:
            model = multi_gpu_model(model,gpus=gpus)
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return model,base

def download(filename):
    import pandas_datareader.data as pdr
    import yfinance as yf
    import datetime as dt
    yf.pdr_override()

    today = '{0:%Y-%m-%d}'.format(dt.date.today())
    data = pdr.get_data_yahoo('^N225','2000-01-01',)
    data.to_csv(filename)



if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-l','--learn',action='store_true')
    parser.add_argument('-v','--visualize',action='store_true')
    parser.add_argument('-f','--csv_filename',type=str,default='^N225.csv')
    parser.add_argument('-a','--compare_all',action='store_true')
    parser.add_argument('-g','--gpus',type=int,default=1)
    parser.add_argument('-u','--update_csv',action='store_true')
    parser.add_argument('-q','--qrnn',action='store_true')
    args = parser.parse_args()

    if(args.update_csv):
        download(args.csv_filename)

    if(args.qrnn):
        a=KabuQRNN(filename=args.csv_filename)
    else:
        a=KabuLSTM(filename=args.csv_filename)

    a._read()
    if(args.learn):
        x,y,z = a._generate()
        model,base = a._build(gpus=args.gpus)
        base.summary()
        a._calculate(model,x,y)
        a._predict(model,z)
        a._save(base)
    elif(args.visualize):
        from keras.utils import plot_model
        model,base = a._build(gpus=args.gpus)
        a._load(model)
        base.summary()
        plot_model(base, to_file='model.png')
    elif(args.compare_all):
        x,y,z = a._generate()
        model,base = a._build(gpus=args.gpus)
        a._load(model)
        a._validate(model,x,y)
    else:
        x,y,z = a._generate()
        model,base = a._build(gpus=args.gpus)
        a._load(model)
        a._predict(model,z)
