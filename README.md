# RealTimeStockMarket

## Structure
This Project consists of two parts<br>
1: **Flask API**       : This is responsible for excecuting the model predictions<br>
2: **Node.js Web App** : This is responsible for rendering the Dashboard Pages [*Just Because I wanted to practice Node.js, you can do it using Flask too*]

## Definition
We have integrated Different types of Deep Learning Networks in this Repository.<br>
The models we have implemented are as follows : <br>
* 1D CNN+LSTM      RMSE : 4.53112	MAE : 3.40922<br>
* 1D CNN+GRU       RMSE : 3.79188	MAE : 2.1223
* LSTM             RMSE : 4.84941	MAE : 3.03043
* GRU              RMSE : 8.4643	MAE : 7.6212
* 1D CNN           RMSE : 3.79188	MAE : 2.1223

## Usage
For the Flask part, refer to [this](Flask\API/README.md) 

For the Node.js part, refer to [this](WEBAPP/README.md)