import os
import keras
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from alpha_vantage.timeseries import TimeSeries

def featureScaling(training_set):
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    return training_set_scaled,sc

def retrain(train_x, train_y,name):
    train_x, train_y = np.array(train_x), np.array(train_y)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

    model = load_model(name)
    print(f"......GO TO SLEEP ITS {name}................")
    model.fit(train_x,train_y,epochs=50,batch_size=32)
    print(f"......GET UP ITS {name}!!!!!!!!!!!!!!!")
    model.save(name)
    return model

def last100values(company):
    company_symbol = 'NSE:'+ company
    ts = TimeSeries(key='XAYE00R4HWQOB5SU', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=company_symbol,interval='15min', outputsize='compact')
    training_set = data.iloc[:, 1:2].values
    return training_set

#Have to add time series ka thing last one day ka value

# dataset_train = pd.read_csv("/content/INFY__EQ__NSE__NSE__15__MINUTE.csv")
# dataset_train["timestamp"] = pd.to_datetime(dataset_train["timestamp"])
# dataset_train.dropna(axis = 0, how ='any',inplace=True)

# training_set = dataset_train.iloc[:, 2:3].values
training_set = last100values('INFY')
training_set = training_set[::-1]

training_set_scaled,sc = featureScaling(training_set)

train_x =[]
train_y =[]
for i in range(40, len(training_set_scaled)):
    train_x.append(training_set_scaled[i-40:i, 0])
    train_y.append(training_set_scaled[i, 0])


allModels = ['cnnlstm.h5','cnngru.h5','GRU.h5','LSTM.h5','cnn.h5']
trained_models = []

for name in allModels:
    trained_models.append(retrain(train_x, train_y,name))
