#! /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import scipy as sp
import scipy.fftpack

from keras.models import Sequential, model_from_json, Model
from keras.layers import Dense, Activation, Dropout, InputLayer, Bidirectional, Input, Multiply, Concatenate
from keras.layers.recurrent import LSTM, RNN, SimpleRNN, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

class Kabu:
    def __init__(self,filename='^N225.csv'):
        self._data =[]
        self._filename = filename
        self._config = {
            'days':4000,
            'keep':25,
            'term':64,
            #'category':(-.3,.0,+.3)
            'category':(-.07,-.03,-.01,-.005,.0,+.005,+.01,+.03,+.07),
            }
        self._ml = {'hidden':500,'epoch':200,'batch':64}
        self._x = []
        self._y = []
        self._z = []
        self._wx = []
        self._wz = []

    def _read(self):
        self._data = pd.read_csv(self._filename,index_col=0)
        #self._data = self._data[self._data.Volume>0.]
        self._data = self._data.drop('Volume',axis=1)
        self._data = self._data.dropna(how='any')
        self._data = np.log(self._data)[-self._config['days']:]

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

    def _rule2(self,data,counts=6):
        diff = []
        for k in data.index:
            #翌日購入,翌々日売却
            buy = data.at[k,(1,'Open')]
            sell = data.at[k,(2,'Open')]
            diff.append(sell-buy)
        separator = pd.qcut(diff,counts)
        '''
        s_diff = sorted(diff)
        separator = [s_diff[x*int(len(data)/counts)]
            for x in range(1,counts)]
        '''
        np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
<<<<<<< HEAD
        print(100.*(np.exp(np.array(separator.value))-1.))
=======
        print(100.*(np.exp(np.array(separator.values))-1.))
>>>>>>> 691abc867035abb79d0fbd9d897c260edc07fb03

        output = []
        for d in diff:
            #翌日購入,翌々日売却
            category = np.zeros(counts)
            for j,theta in enumerate(separator):
                if d < theta:
                    category[j] = 1.
                    break
            else:
                category[counts-1] = 1.
            assert sum(category)==1, '条件漏れ'
            output.append(category)

        output = pd.DataFrame(output,index=data.index)

        return output

    def _rule3(self,data):
        counts=3
        output = []
        for k in data.index:
            #翌日購入,翌々日売却
            buy = data.at[k,(1,'Open')]
            category = np.zeros(counts)
            for x in range(2,self._config['keep']):
                if data.at[k,(x,'Low')] - buy  < np.log(1.-.03):
                    category[2]=1.
                    break
                elif data.at[k,(x,'Open')] - buy >= np.log(1.+.05):
                    category[1]=1.
                    break
            else:
                category[0]=1.
            output.append(category)

        output = pd.DataFrame(output,index=data.index)
        return output

    def _generate(self):
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
            before.values.flatten().reshape(-1,1),
            #scaler.fit_transform(before.values.flatten().reshape(-1,1)),
            [len(before.index), self._config['term'], len(self._data.columns)])
        label = self._rule2(after)
        dataset2 = np.reshape(
            before.sort_index(axis=1,level=1).values.flatten().reshape(-1,1),
            #scaler.fit_transform(before.sort_index(axis=1,level=1).values.flatten().reshape(-1,1)),
            [len(before.index), len(self._data.columns), self._config['term']])
        #離散コサイン変換
        wave = np.abs(sp.fftpack.fft(dataset2,axis=2))
        #wave = np.power(sp.fftpack.dct(dataset2,axis=2),2.)

        self._y = label.values
        self._x,self._z = np.split(dataset,[len(self._y)])
        self._wx,self._wz = np.split(wave,[len(self._y)])

    def _build(self):
        days = self._config['term']
        dimension = len(self._data.columns)

        input_raw = Input(shape=(days,dimension))
        drop_a1 = Dropout(.2)(input_raw)
        lstm_a = Bidirectional(GRU(
        #lstm_a = Bidirectional(LSTM(
            self._ml['hidden'],
            use_bias=True,
            #return_sequences=False,
            return_sequences=True,
            input_shape=(days, dimension),
            activation='relu'))(drop_a1)
        lstm_a2 = Bidirectional(GRU(
            self._ml['hidden'],
        ))(lstm_a)
        drop_a2 = Dropout(.5)(lstm_a2)
        #drop_a2 = Dropout(.5)(lstm_a)

        input_wav = Input(shape=(dimension,days))
        drop_b1 = Dropout(.2)(input_wav)
        lstm_b = Bidirectional(GRU(
        #lstm_b = Bidirectional(LSTM(
            self._ml['hidden'],
            use_bias=True,
            #return_sequences=False,
            return_sequences=True,
            input_shape=(dimension, days),
            activation='relu'))(drop_b1)
        lstm_b2 = Bidirectional(GRU(
            self._ml['hidden'],
        ))(lstm_b)
        drop_b2 = Dropout(.5)(lstm_b2)
        #drop_b2 = Dropout(.5)(lstm_b)

        merged = Concatenate()([drop_a2,drop_b2])
        '''
        lstm_2 = Bidirectional(GRU(
            self._ml['hidden']
        ))(merged)
        dense_1 = Dense(5000)(lstm_2)
        '''
        dense_1 = Dense(5000)(merged)
        dense_2 = Dense(
            len(self._y[0]),
            kernel_initializer='glorot_uniform')(dense_1)
        output = Activation('softmax')(dense_2)

        model = Model(inputs=[input_raw,input_wav],outputs=output)
        optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self._model = model

    def _calculate(self):
        early_stopping = EarlyStopping(patience=5, verbose=1)
        self._model.fit(
            [self._x,self._wx], self._y,
            epochs=self._ml['epoch'],
            batch_size=self._ml['batch'],
            validation_split=0.2,
            shuffle=False,
            callbacks=[early_stopping])

    def _predict(self):
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        ans = self._model.predict([self._z,self._wz])
        print(np.round(ans,decimals=2))

    def _validate(self):
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        ans = self._model.predict([self._x,self._wx])
        ans = list(zip(self._y,ans))
        for input,output in np.round(ans,decimals=2):
            print(input,output,'=>',np.dot(input,output))

if __name__ == '__main__':
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-l','--learn',action='store_true')
    parser.add_argument('-v','--visualize',action='store_true')
    parser.add_argument('-f','--csv_filename',type=str,default='^N225.csv')
    parser.add_argument('-a','--compare_all',action='store_true')
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
        a._build()
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
