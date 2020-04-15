import json
import warnings
import numpy as np
from flask import Flask
from flask import jsonify
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


app=Flask(__name__)


#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
from alpha_vantage.timeseries import TimeSeries
def last100values(company):
  company_symbol = 'NSE:'+ company
  ts = TimeSeries(key='XAYE00R4HWQOB5SU', output_format='pandas')
  data, meta_data = ts.get_intraday(symbol=company_symbol,interval='15min', outputsize='combat')
  training_set = data.iloc[:, 1:2].values
  return training_set

def getNewValue(latest_data,name):
    model = load_model('WeightFiles/'+name)
    
    xxxx = latest_data[0:40][::-1]

    sc = MinMaxScaler(feature_range=(0,1))
    x_scaled = sc.fit_transform(xxxx)

    latest_predict_x = np.array(x_scaled)
    latest_predict_x = np.reshape(latest_predict_x, (1,latest_predict_x.shape[0], latest_predict_x.shape[1]))

    print(latest_predict_x.shape)

    #predicting the next value
    latest_predicted_y = model.predict(latest_predict_x)

    #reverse feature scaling
    predicted_stock_price = sc.inverse_transform(latest_predicted_y)

    return predicted_stock_price





#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#

@app.route("/api/getall/:company")
def home():
    company = 'INFY'
    allModels = ['cnnlstm.h5']
    allPredictions = []

    latest_data = last100values(company)

    for i in allModels:
        predicted = getNewValue(latest_data,i)
        allPredictions.append(predicted)

    toSendSeq = latest_data[::-1]
    flat_list = [item for sublist in toSendSeq for item in sublist]

    needed = allPredictions[0][0].tolist()
    
    myAll = {   
                'sequence' : flat_list,
                'CNNLSTM' : needed[0]
            }
    print(myAll)

    return json.dumps(myAll)



if __name__ == "__main__":
    app.run(debug=True)