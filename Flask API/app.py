import json
import warnings
import numpy as np
from flask import Flask
from flask import jsonify
from flask_cors import CORS 
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

KEY = ''#add Your Key
app=Flask(__name__)
CORS(app)

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
from alpha_vantage.timeseries import TimeSeries

def getNewValue(latest_data,name):
    model = load_model('WeightFiles/'+name)
    
    xxxx = latest_data[0:60][::-1]

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



def getMinDataTo15Data(company):
    company_symbol = 'NSE:'+ company
    ts = TimeSeries(key=KEY, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=company_symbol,interval='1min', outputsize='full')

    training_set = data.iloc[:900, 0:1]

    indices=[]
    for i in range(len(training_set)):
        if i%15 != 0:
            indices.append(i)

    new_dataset = training_set.reset_index()
    made_dataset = new_dataset.drop(indices)

    final_dataset = made_dataset.reset_index()
    main_dataset = final_dataset.drop('index',axis=1)

    training_set_values = main_dataset.iloc[:,1:2].values
    dates = main_dataset.iloc[:,0].tolist()
    stocks = main_dataset.iloc[:,1].tolist()

    from datetime import datetime
    string_dates =[]
    for date in dates: 
        string_dates.append(date.strftime("%m-%d-%Y-%H-%M"))
    


    return string_dates,stocks,training_set_values

def getDetailedMinData(company):
    company_symbol = 'NSE:'+ company
    ts = TimeSeries(key=KEY, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=company_symbol,interval='1min', outputsize='compact')


    indices=[]
    training_set = data.iloc[:60, 0:1]
    new_dataset = training_set.reset_index()
    made_dataset = new_dataset.drop(indices)

    final_dataset = made_dataset.reset_index()
    main_dataset = final_dataset.drop('index',axis=1)

    dates = main_dataset.iloc[:,0].tolist()
    stocks = main_dataset.iloc[:,1].tolist()

    nextTime = dates[0] + timedelta(minutes = 15)
    nextTime = nextTime.strftime("%m-%d-%Y-%H-%M")

    string_dates =[]
    for date in dates: 
        string_dates.append(date.strftime("%m-%d-%Y-%H-%M"))

    string_dates = string_dates[::-1]
    stocks = stocks[::-1]

    return string_dates,stocks,nextTime


#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#

@app.route("/api/getall/<name>", methods = ['GET'])
def home(name):
    company = name
   
    allModels = ['cnnlstm.h5','cnngru.h5','GRU.h5','LSTM.h5','cnn.h5']
    allPredictions = []

    dates, stocks , latest_data = getMinDataTo15Data(company)
    original15Data = latest_data[0:60][::-1]

    for i in allModels:
        predicted, = getNewValue(latest_data,i)
        allPredictions.append(predicted)

    flat_list = [item for sublist in original15Data for item in sublist]

    neededCL = allPredictions[0][0].tolist()
    neededCG = allPredictions[1][0].tolist()
    neededG = allPredictions[2][0].tolist()
    neededL = allPredictions[3][0].tolist()
    neededC = allPredictions[4][0].tolist()

    list_CL = flat_list
    list_CL.append(neededCL)

    list_CG = flat_list
    list_CG.append(neededCG)

    list_G = flat_list
    list_G.append(neededG)

    list_L = flat_list
    list_L.append(neededL)

    list_C = flat_list
    list_C.append(neededC)
    
    minDates , minStocks , nextTime = getDetailedMinData(company)


    myAll = {   
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Dates': nextTime,
                'CNNLSTM'   : neededCL,#done
                'CNNGRU'    : neededCG,#done
                'GRU'       : neededG,#done
                'LSTM'      : neededL,#done
                'CNN'       : neededC#done
            }

    print(myAll)

    return json.dumps(myAll)



@app.route("/api/cnn/<name>", methods = ['GET'])
def cnn(name):
    company = name
   
    allModels = ['cnn.h5']
    allPredictions = []

    dates, stocks , latest_data = getMinDataTo15Data(company)
    c1Real = latest_data[0]
    c2Real = latest_data[1]
    data_for_one = latest_data[1:61]
    data_for_two = latest_data[2:62]

    testing_models = ['cnnlstm.h5','lstm.h5']

    cnnlstm_prediction = []
    lstm_prediction = []

    cnnlstm_predicted_one = getNewValue(data_for_one,testing_models[0])[0].tolist()[0] #cnnlstm----1
    cnnlstm_predicted_two = getNewValue(data_for_two,testing_models[0])[0].tolist()[0] #cnnlstm----2

    lstm_predicted_one = getNewValue(data_for_one,testing_models[1])[0].tolist()[0] #lstm----1
    lstm_predicted_two = getNewValue(data_for_two,testing_models[1])[0].tolist()[0] #lstm----2

    #cnnlstm_1*x + lstm_1*y  = c1Real
    #cnnlstm_2*x + lstm_2*y  = c2Real

    X = np.array([ [cnnlstm_predicted_one,lstm_predicted_one],
                   [cnnlstm_predicted_two,lstm_predicted_two] ])

    Y = np.array( [ [c1Real,c2Real] ] )

    solved_array = np.linalg.solve(X,Y)


    cnnlstm_weight = solved_array[0]
    lstm_weight = solved_array[1]

    #predicting cnnlstm
    cnnlstm_answer = getNewValue(latest_data,'cnnlstm.h5')
    lstm_answer = getNewValue(latest_data,'lstm.h5')

    true_value = []
    true_value.append(   ( (lstm_weight)*(lstm_answer[0].tolist()[0]) + (cnnlstm_weight)*(cnnlstm_answer[0].tolist()[0]) )    )

    original15Data = latest_data[0:60][::-1]

    for i in allModels:
        predicted, = getNewValue(latest_data,i)
        allPredictions.append(predicted)

    flat_list = [item for sublist in original15Data for item in sublist]

    neededC = allPredictions[0][0].tolist()

    list_C = flat_list
    list_C.append(neededC)
    
    minDates , minStocks , nextTime = getDetailedMinData(company)


    myAll = {   
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Date' : nextTime,
                'CNN'       : neededC#done  #put true_value
            }

    print(myAll)

    return json.dumps(myAll)


@app.route("/api/cnnlstm/<name>", methods = ['GET'])
def cnnlstm(name):
    company = name
   
    allModels = ['cnnlstm.h5']
    allPredictions = []

    dates, stocks , latest_data = getMinDataTo15Data(company)
    original15Data = latest_data[0:60][::-1]

    for i in allModels:
        predicted, = getNewValue(latest_data,i)
        allPredictions.append(predicted)

    flat_list = [item for sublist in original15Data for item in sublist]

    neededCL = allPredictions[0][0].tolist()

    list_CL = flat_list
    list_CL.append(neededCL)
    
    minDates , minStocks , nextTime = getDetailedMinData(company)


    myAll = {   
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Date': nextTime,
                'CNNLSTM'   : neededCL#done
            }

    print(myAll)

    return json.dumps(myAll)

@app.route("/api/cnngru/<name>", methods = ['GET'])
def cnngru(name):
    company = name
   
    allModels = ['cnngru.h5']
    allPredictions = []

    dates, stocks , latest_data = getMinDataTo15Data(company)
    original15Data = latest_data[0:60][::-1]

    for i in allModels:
        predicted, = getNewValue(latest_data,i)
        allPredictions.append(predicted)

    flat_list = [item for sublist in original15Data for item in sublist]

    neededCG = allPredictions[0][0].tolist()

    list_CG = flat_list
    list_CG.append(neededCG)
    
    minDates , minStocks , nextTime = getDetailedMinData(company)

    myAll = {   
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Date' : nextTime,
                'CNNGRU'   : neededCG#done
            }

    return json.dumps(myAll)

@app.route("/api/lstm/<name>", methods = ['GET'])
def lstm(name):
    company = name
   
    allModels = ['LSTM.h5']
    allPredictions = []

    dates, stocks , latest_data = getMinDataTo15Data(company)
    original15Data = latest_data[0:60][::-1]

    for i in allModels:
        predicted, = getNewValue(latest_data,i)
        allPredictions.append(predicted)

    flat_list = [item for sublist in original15Data for item in sublist]

    neededL = allPredictions[0][0].tolist()

    list_L = flat_list
    list_L.append(neededL)
    
    minDates , minStocks , nextTime = getDetailedMinData(company)

    myAll = {   
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Date' : nextTime,
                'LSTM'   : neededL#done
            }

    return json.dumps(myAll)

@app.route("/api/gru/<name>", methods = ['GET'])
def gru(name):
    company = name
   
    allModels = ['GRU.h5']
    allPredictions = []

    dates, stocks , latest_data = getMinDataTo15Data(company)
    original15Data = latest_data[0:60][::-1]

    for i in allModels:
        predicted, = getNewValue(latest_data,i)
        allPredictions.append(predicted)

    flat_list = [item for sublist in original15Data for item in sublist]

    neededG = allPredictions[0][0].tolist()

    list_G = flat_list
    list_G.append(neededG)
    
    minDates , minStocks , nextTime = getDetailedMinData(company)

    myAll = {   
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Date' : nextTime,
                'GRU'   : neededG#done
            }

    return json.dumps(myAll)


@app.route("/api/<name>", methods = ['GET'])
def abc(name):
    company = name
    minDates , minStocks , nextTime = getDetailedMinData(company)

    myAll = {
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Date' : nextTime,
            }
    print(myAll)

    return json.dumps(myAll)


if __name__ == "__main__":
    app.run(debug=True)


