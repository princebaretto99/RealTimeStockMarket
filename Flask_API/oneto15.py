from alpha_vantage.timeseries import TimeSeries
ts = TimeSeries(key='3IV6H0ADFCU3X5UQ', output_format='pandas')
data, meta_data = ts.get_intraday(symbol='NSE:INFY',interval='1min', outputsize='full')

indices=[]
for i in range(len(training_set)):
    if i%15 != 0:
        indices.append(i)


training_set = data.iloc[:600, 0:1]
new_dataset = training_set.reset_index()
# new_dataset = new_dataset.drop('date', axis = 1)
made_dataset = new_dataset.drop(indices)

final_dataset = made_dataset.reset_index()
main_dataset = final_dataset.drop('index',axis=1)

dates = main_dataset.iloc[:,0].tolist()
stocks = main_dataset.iloc[:,1].tolist()