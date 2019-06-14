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
import scipy.stats as ss

class KabuQRNN:
    def __init__(self,filename='^N225.csv',gpus=1):
        self._data =[]
        self._filename = filename
        self._config = {
            'days':500,
            'keep':2,
            'term':64,
            'predict':20,
            #'category':(-.3,.0,+.3)
            'category':(-.07,-.03,-.01,-.005,.0,+.005,+.01,+.03,+.07),
            }
        self._ml = {'hidden':128,'epoch':2000,'batch':1024}
        self._scaler = MinMaxScaler(feature_range=(-1, 1))
        #self._scaler = PowerTransformer(method='box-cox',standardize=True)
        #self._scaler = FunctionTransformer(func=lambda x:x, inverse_func=lambda x:x)
        self._gpus = gpus

    def _read(self):
        self._data = pd.read_csv(self._filename,index_col=0)

        #計算に利用しない列を削除
        self._data = self._data.drop('Volume',axis=1)
        self._data = self._data.drop('Adj Close',axis=1)
        '''
        self._data = self._data.drop('Close',axis=1)
        self._data = self._data.drop('High',axis=1)
        self._data = self._data.drop('Low',axis=1)
        self._data = self._data.drop('Adj Close',axis=1)
        '''

        #列名を番号にする(後でソートしても順番を維持するため)前に退避する
        self._columns = self._data.columns.values
        self._data.columns = range(len(self._data.columns))

        #使えない&使わない行を削除
        self._data = self._data.dropna(how='any')
        self._data = self._data[-self._config['days']:]

        #データを整形
        data = pd.DataFrame(self._scaler.fit_transform(self._data.values),
            index=self._data.index, columns=self._data.columns)
        return data

    def _save(self,model):
        model.save(self._filename+'.h5')

    def _load(self,model):
        model.load_weights(self._filename+'.h5')

    def _generate(self, data):
        term = self._config['term']
        keep = self._config['keep']

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
        #wave = np.abs(sp.fftpack.fft(dataset2,axis=2))

        #離散コサイン変換
        #wave = sp.fftpack.dct(dataset2,axis=2)
        #wave = wave / float(wave.shape[2])

        #離散Wavelet変換
        import pywt
        wave = pywt.wavedec(dataset2, wavelet='haar', axis=2)
        wave = np.concatenate(wave,axis=2)

        y = after.values
        rx,rz = np.split(dataset,[len(y)])
        wx,wz = np.split(wave,[len(y)])

        return [rx,wx],y,[rz,wz]

    def _objective(self,x,y,trial):
        layer_r = trial.suggest_int('layer_r',1,10)
        layer_w = trial.suggest_int('layer_w',1,10)
        hidden = trial.suggest_int('hidden',64,256)
        dropout_rate = trial.suggest_uniform('dropout_rate',0,1)
        activation = trial.suggest_categorical('activation',['sigmoid','relu'])
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'adamax', 'nadam'])
        batch_size = trial.suggest_int('batch_size', 512, 512)

        model, base = self._build(
            layers=[layer_r,layer_w],
            hidden=hidden,
            activation=activation,
            optimizer=optimizer,
            dropout_rate=dropout_rate,
            )
        history = self._calculate(model,x,y,batch_size=batch_size)
        return np.amin(history.history['val_loss'])


    def _build(self, layers=[4,4], hidden=128, activation='sigmoid', optimizer='adam', dropout_rate=0.2):
        days = self._config['term']
        dimension = len(self._data.columns)
        window=2

        input_raw = Input(shape=(days,dimension))
        x = input_raw
        for k in range(layers[0]):
            if k != layers[0]-1:
                s = True
            else:
                s = False
            x = SpatialDropout1D(dropout_rate)(x)
            x = QRNN(
                units=hidden,
                window_size=window,
                return_sequences=s,
                stride=1,
                )(x)

        input_wav = Input(shape=(dimension,days))
        y = input_wav
        for k in range(layers[1]):
            if k != layers[1]-1:
                s = True
            else:
                s = False
            y = SpatialDropout1D(dropout_rate)(y)
            y = QRNN(
                units=hidden,
                window_size=window,
                return_sequences=s,
                stride=1,
                )(y)

        merged = Concatenate()([x,y])
        label = Dense( units= dimension )(merged)
        output = Activation(activation)(label)

        model = Model(inputs=[input_raw,input_wav],outputs=output)
        base = model
        if self._gpus>1:
            model = multi_gpu_model(model,gpus=self._gpus)
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return model,base

    def _calculate(self,model,x,y,batch_size=512):
        early_stopping = EarlyStopping(patience=50, verbose=1)
        history = model.fit(
            x, y,
            epochs=self._ml['epoch'],
            batch_size=batch_size,
            validation_split=0.05,
            shuffle=False,
            callbacks=[early_stopping])
        return history

    def _predict(self,model,data):
        _data = data[-1-self._config['term']:]
        ans = np.zeros((0,len(data.columns)))
        for x in range(self._config['predict']):
            x,y,z = self._generate(_data)
            y = model.predict(z)
            ans = np.append(ans,y,axis=0)
            _data = data.append(pd.DataFrame(y,columns=data.columns))

        ans = np.array(ans)
        ans = self._scaler.inverse_transform(ans)
        print(np.round(ans,decimals=2))

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
    def _build(self, layers=[4,4], hidden=128, activation='sigmoid', optimizer='adam', dropout_rate=0.2):
        days = self._config['term']
        dimension = len(self._data.columns)

        input_raw = Input(shape=(days,dimension))
        x = input_raw
        for k in range(layers[0]):
            if k != layers[0]-1:
                s = True
            else:
                s = False
            x = Dropout(dropout_rate)(x)
            x = Bidirectional(LSTM(
                units=hidden,
                return_sequences=s,
                ))(x)

        input_wav = Input(shape=(dimension,days))
        y = input_wav
        for k in range(layers[1]):
            if k != layers[1]-1:
                s = True
            else:
                s = False
            y = Dropout(dropout_rate)(y)
            y = Bidirectional(LSTM(
                units=hidden,
                return_sequences=s,
                ))(y)

        merged = Concatenate()([x,y])
        label = Dense( units= dimension )(merged)
        output = Activation(activation)(label)

        model = Model(inputs=[input_raw,input_wav],outputs=output)
        base = model
        if self._gpus>1:
            model = multi_gpu_model(model,gpus=self._gpus)
        model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        return model,base

def download(filename,code='^N225'):
    import pandas_datareader.data as pdr
    import yfinance as yf
    import datetime as dt
    yf.pdr_override()

    today = '{0:%Y-%m-%d}'.format(dt.date.today())
    data = pdr.get_data_yahoo(code,'2000-01-01',)
    data.to_csv(filename)



if __name__ == '__main__':
    import json
    json_filename = 'param.json'

    #コマンド引数
    import argparse as ap
    parser = ap.ArgumentParser()
    parser.add_argument('-l','--learn',action='store_true')
    parser.add_argument('-v','--visualize',action='store_true')
    #parser.add_argument('-f','--csv_filename',type=str,default='^N225.csv')
    parser.add_argument('-c','--code',type=str,default='^N225')
    parser.add_argument('-a','--compare_all',action='store_true')
    parser.add_argument('-g','--gpus',type=int,default=1)
    parser.add_argument('-u','--update_csv',action='store_true')
    parser.add_argument('-q','--qrnn',action='store_true')
    parser.add_argument('-o','--optimize',nargs='?',type=int,const=1,default=0)
    args = parser.parse_args()

    #CSVファイル名
    csv_filename = args.code+'.csv'

    #最新株価ダウンロード
    if(args.update_csv):
        download(csv_filename,arg.code)

    #パラメタ設定ファイルの読み込み
    try:
        with open(json_filename,'r') as fp:
            parameters = json.load(fp)
    except IOError:
        parameters = {}

    #計算インスタンス作成
    if(args.qrnn):
        name = 'QRNN'
        a=KabuQRNN(filename=csv_filename,gpus=args.gpus)
    else:
        name = 'LSTM'
        a=KabuLSTM(filename=csv_filename,gpus=args.gpus)

    #データ準備
    data = a._read()

    #学習
    if(args.learn):
        x,y,z = a._generate(data)
        model,base = a._build(**parameters[name]['model'])
        base.summary()
        a._calculate(model,x,y,**parameters[name]['learning'])
        a._save(base)

    #ハイパーパラメタ最適化
    elif(args.optimize>0):
        import optuna, functools
        x,y,z = a._generate(data)
        f = functools.partial(a._objective,x,y)

        db_name = 'study.db'
        try:
            study = optuna.load_study(study_name=name,storage='sqlite:///'+db_name)
        except ValueError:
            study = optuna.create_study(study_name=name,storage='sqlite:///'+db_name,direction='minimize')
        study.optimize(f,n_trials=args.optimize)

        best = study.best_params
        parameters[name] = {
            'model':{
                'layers':[best['layer_r'],best['layer_w']],
                'hidden':best['hidden'],
                'activation':best['activation'],
                'optimizer':best['optimizer'],
                'dropout_rate':best['dropout_rate'],
            },
            'learning':{
                'batch_size':best['batch_size'],
            },
        }
        with open(json_filename,'w') as fp:
            json.dump(parameters,fp,indent=4)

    #過去データとの比較
    elif(args.compare_all):
        x,y,z = a._generate(data)
        model,base = a._build(**parameters[name]['model'])
        a._load(model)
        a._validate(model,x,y)

    #モデル出力
    elif(args.visualize):
        from keras.utils import plot_model
        model,base = a._build(**parameters[name]['model'])
        a._load(model)
        base.summary()
        plot_model(base, to_file='model.png')

    #予測
    else:
        x,y,z = a._generate(data)
        model,base = a._build(**parameters[name]['model'])
        a._load(model)
        a._predict(model,data)
