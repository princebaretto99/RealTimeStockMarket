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
# def last100values(company):
#     company_symbol = 'NSE:'+ company
#     ts = TimeSeries(key='XAYE00R4HWQOB5SU', output_format='pandas')
#     data, meta_data = ts.get_intraday(symbol=company_symbol,interval='15min', outputsize='compact')

#     forDate = data.iloc[:, 0:1]
#     new_dataset = forDate.reset_index()
#     dates = new_dataset.iloc[:,0].tolist()
#     dates = dates[::-1]

#     training_set = data.iloc[:, 0:1].values
#     return training_set,dates

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



def getMinDataTo15Data(company):
    company_symbol = 'NSE:'+ company
    ts = TimeSeries(key='3IV6H0ADFCU3X5UQ', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=company_symbol,interval='1min', outputsize='full')

    training_set = data.iloc[:600, 0:1]

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
        string_dates.append(date.strftime("%m/%d/%Y %H:%M"))
    


    return string_dates,stocks,training_set_values

def getDetailedMinData(company):
    company_symbol = 'NSE:'+ company
    ts = TimeSeries(key='3IV6H0ADFCU3X5UQ', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=company_symbol,interval='1min', outputsize='compact')


    indices=[]
    training_set = data.iloc[:40, 0:1]
    new_dataset = training_set.reset_index()
    made_dataset = new_dataset.drop(indices)

    final_dataset = made_dataset.reset_index()
    main_dataset = final_dataset.drop('index',axis=1)

    dates = main_dataset.iloc[:,0].tolist()
    stocks = main_dataset.iloc[:,1].tolist()

    string_dates =[]
    for date in dates: 
        string_dates.append(date.strftime("%m/%d/%Y %H:%M"))

    string_dates = string_dates[::-1]
    stocks = stocks[::-1]

    return string_dates,stocks


#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#

@app.route("/api/getall/<name>", methods = ['GET'])
def home(name):
    company = name
   
    allModels = ['cnnlstm.h5','cnngru.h5','GRU.h5','LSTM.h5','cnn.h5']
    allPredictions = []

    dates, stocks , latest_data = getMinDataTo15Data(company)
    original15Data = latest_data[0:40][::-1]

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
    
    minDates , minStocks = getDetailedMinData(company)


    myAll = {   
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Dates': dates,
                'CNNLSTM'   : list_CL,#done
                'CNNGRU'    : list_CG,#done
                'GRU'       : list_G,#done
                'LSTM'      : list_L,#done
                'CNN'       : list_C#done
            }

    print(myAll)

    return json.dumps(myAll)



@app.route("/api/cnn/<name>", methods = ['GET'])
def cnn(name):
    company = name
   
    allModels = ['cnn.h5']
    allPredictions = []

    dates, stocks , latest_data = getMinDataTo15Data(company)
    original15Data = latest_data[0:40][::-1]

    for i in allModels:
        predicted, = getNewValue(latest_data,i)
        allPredictions.append(predicted)

    flat_list = [item for sublist in original15Data for item in sublist]

    neededC = allPredictions[0][0].tolist()

    list_C = flat_list
    list_C.append(neededC)
    
    minDates , minStocks = getDetailedMinData(company)


    myAll = {   
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Dates': dates,
                'CNN'       : list_C#done
            }

    print(myAll)

    return json.dumps(myAll)


@app.route("/api/cnnlstm/<name>", methods = ['GET'])
def cnnlstm(name):
    company = name
   
    allModels = ['cnnlstm.h5']
    allPredictions = []

    dates, stocks , latest_data = getMinDataTo15Data(company)
    original15Data = latest_data[0:40][::-1]

    for i in allModels:
        predicted, = getNewValue(latest_data,i)
        allPredictions.append(predicted)

    flat_list = [item for sublist in original15Data for item in sublist]

    neededCL = allPredictions[0][0].tolist()

    list_CL = flat_list
    list_CL.append(neededCL)
    
    minDates , minStocks = getDetailedMinData(company)


    myAll = {   
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Dates': dates,
                'CNNLSTM'   : list_CL#done
            }

    print(myAll)

    return json.dumps(myAll)

@app.route("/api/cnngru/<name>", methods = ['GET'])
def cnngru(name):
    company = name
   
    allModels = ['cnngru.h5']
    allPredictions = []

    dates, stocks , latest_data = getMinDataTo15Data(company)
    original15Data = latest_data[0:40][::-1]

    for i in allModels:
        predicted, = getNewValue(latest_data,i)
        allPredictions.append(predicted)

    flat_list = [item for sublist in original15Data for item in sublist]

    neededCG = allPredictions[0][0].tolist()

    list_CG = flat_list
    list_CG.append(neededCG)
    
    minDates , minStocks = getDetailedMinData(company)

    myAll = {   
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Dates': dates,
                'CNNLSTM'   : list_CG#done
            }

    return json.dumps(myAll)

@app.route("/api/lstm/<name>", methods = ['GET'])
def lstm(name):
    company = name
   
    allModels = ['LSTM.h5']
    allPredictions = []

    dates, stocks , latest_data = getMinDataTo15Data(company)
    original15Data = latest_data[0:40][::-1]

    for i in allModels:
        predicted, = getNewValue(latest_data,i)
        allPredictions.append(predicted)

    flat_list = [item for sublist in original15Data for item in sublist]

    neededL = allPredictions[0][0].tolist()

    list_L = flat_list
    list_L.append(neededL)
    
    minDates , minStocks = getDetailedMinData(company)

    myAll = {   
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Dates': dates,
                'CNNLSTM'   : list_L#done
            }

    return json.dumps(myAll)

@app.route("/api/gru/<name>", methods = ['GET'])
def gru(name):
    company = name
   
    allModels = ['GRU.h5']
    allPredictions = []

    dates, stocks , latest_data = getMinDataTo15Data(company)
    original15Data = latest_data[0:40][::-1]

    for i in allModels:
        predicted, = getNewValue(latest_data,i)
        allPredictions.append(predicted)

    flat_list = [item for sublist in original15Data for item in sublist]

    neededG = allPredictions[0][0].tolist()

    list_G = flat_list
    list_G.append(neededG)
    
    minDates , minStocks = getDetailedMinData(company)

    myAll = {   
                'minDates'  : minDates,#done
                'minStocks' : minStocks,#done
                'min15Dates': dates,
                'CNNLSTM'   : list_G#done
            }

    return json.dumps(myAll)


@app.route("/api/getall", methods = ['GET'])
def abc():
    myAll = {
        'sequence' : "list",
        'CNNLSTM' : 'carol',
        'CNNGRU' : 123
    }
    print(myAll)

    return json.dumps(myAll)


if __name__ == "__main__":
    app.run(debug=True)




# cors error = "https://medium.com/@dtkatz/3-ways-to-fix-the-cors-error-and-how-access-control-allow-origin-works-d97d55946d9"