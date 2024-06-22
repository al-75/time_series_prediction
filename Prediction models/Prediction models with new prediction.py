import logging
logging.captureWarnings(True)
#import openpyxl
import pandas as pd
import datetime as dt
import numpy as np
from datetime import timedelta, date

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from tabulate import tabulate
import matplotlib.pyplot as plt
#import quandl
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from pandas.plotting import register_matplotlib_converters

import joblib

#pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def code_mean(data, cat_feature, real_feature):
  
    return dict(data.groupby(cat_feature)[real_feature].mean())


def prepareData(data, lag_start, lag_end, test_size):
    
    data.columns = ["y"]

    # считаем индекс в датафрейме, после которого начинается тестовыый отрезок
    
    test_index = int(len(data)*(1-test_size))

    # добавляем лаги исходного ряда в качестве признаков
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.y.shift(i)
   
    
     
    #/В зависимости какая у нас временная шкала - или день недели или месяц
    
    data['weekday']=data.index.month
    data['quarter']=data.index.quarter
   
    # считаем средние только по тренировочной части, чтобы избежать лика
    data['weekday_average'] = data.weekday.map(code_mean(data[:test_index], 'weekday', 'y'))
    data['quarter_average']=data.quarter.map(code_mean(data[:test_index], 'quarter', 'y'))
    # выкидываем закодированные средними признаки 
    data.drop(['weekday','quarter'], axis=1, inplace=True)
    
    data = data.dropna()
    
    
    data = data.reset_index(drop=True)
    
    
    test_index = int(len(data)*(1-test_size))
    # разбиваем весь датасет на тренировочную и тестовую выборку
    
    X_train = data.loc[:test_index].drop(["y"], axis=1)
    y_train = data.loc[:test_index]["y"]
    X_test = data.loc[test_index:].drop(["y"], axis=1)
    y_test = data.loc[test_index:]["y"]

    return X_train, X_test, y_train, y_test



data=pd.read_excel('Prediction model.xlsx',sheet_name='запрос',usecols=('Период','Количество'), parse_dates=['Период'])
#data=data[:-4]
data=data.set_index(['Период'],drop=True)


#Тестируем
X_train, X_test, y_train, y_test = prepareData(data,lag_start=4, lag_end=8,test_size=0.4)
#lr = MLPRegressor(hidden_layer_sizes=(5,5),activation='relu', solver='lbfgs',learning_rate_init=0.3, max_iter=30000)
#lr = LinearRegression()
lr=RandomForestRegressor(n_estimators=100, max_features ='sqrt')
#lr=KNeighborsRegressor(n_neighbors=5)
#lr=LogisticRegression()
#lr=SVR(kernel='linear')

lr.fit(X_train, y_train)

#////////////////////////////////////////
# Загрузка в файл модели
#joblib.dump(lr,'preforma.pkl', compress=9)
#lr= joblib.load('preforma.pkl')#Выгрузка модели из файла


X_test=data[-4:].loc[:,'lag_4':].reset_index(drop=True)
prediction = lr.predict(X_test)
#Классный mapping
f=X_test.index.map(lambda x:data[-4:].loc[:,'lag_4':].index[x])

fig,ax=plt.subplots(1,1,figsize=(10, 5))
#plt.plot(prediction, "ro-", label="prediction")
plt.plot_date(f,prediction, "ro-", label="prediction")


#plt.plot_date(f,y_test.values, linestyle='dashed',label="actual")
plt.legend(loc="best")


plt.show()


