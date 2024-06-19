
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


def code_mean(data, cat_feature, real_feature):
  
    return dict(data.groupby(cat_feature)[real_feature].mean())


def prepareData(data, lag_start, lag_end, test_size):
    
    data.columns = ["y"]

    # считаем индекс в датафрейме, после которого начинается тестовыый отрезок
    
    test_index = int(len(data)*(1-test_size))

    # добавляем лаги исходного ряда в качестве признаков
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.y.shift(i)

    #Добавляем новые признаки времени
    
    data["weekday"]=data.index.month
    data["quarter"]=data.index.quarter
   
    # считаем средние только по тренировочной части
    data['weekday_average'] = data.weekday.map(code_mean(data[:test_index], 'weekday', 'y'))
    data["quarter_average"]=data.quarter.map(code_mean(data[:test_index], 'quarter', 'y'))
    # выкидываем закодированные средними признаки 
    data.drop(["weekday"], axis=1, inplace=True)
    data.drop(["quarter"], axis=1, inplace=True)
    
    data = data.dropna()
    
    data = data.reset_index(drop=True)
    
    test_index = int(len(data)*(1-test_size))

    # разбиваем весь датасет на тренировочную и тестовую выборку
    
    X_train = data.loc[:test_index].drop(["y"], axis=1)
    y_train = data.loc[:test_index]["y"]
    X_test = data.loc[test_index:].drop(["y"], axis=1)
    y_test = data.loc[test_index:]["y"]

    return X_train, X_test, y_train, y_test


#Источник данных
data=pd.read_excel('Prediction model.xlsx',sheet_name='запрос',usecols=('Период','Количество'), parse_dates=['Период'])
data=data.set_index(['Период'],drop=True) 

#data=pd.date_range('2021-04-01',periods=5,freq='M')


#Тестируем
X_train, X_test, y_train, y_test = prepareData(data,lag_start=4, lag_end=8,test_size=0.3)
#lr = MLPRegressor(hidden_layer_sizes=(5,5),activation='relu', solver='lbfgs',learning_rate_init=0.3, max_iter=30000)
#lr = LinearRegression()
lr=RandomForestRegressor(n_estimators=100, max_features ='sqrt')
#lr=KNeighborsRegressor(n_neighbors=5)
#lr=LogisticRegression()
#lr=SVR(kernel='linear')

#lr.fit(X_train, y_train)

#////////////////////////////////////////
# Загрузка в файл модели
#joblib.dump(lr,'preforma.pkl', compress=9)
lr= joblib.load('preforma.pkl')#Выгрузка модели из файла


prediction = lr.predict(X_test)

f=X_test.index.map(lambda x:data.index[x])

fig,ax=plt.subplots(1,1,figsize=(15, 7))
plt.plot_date(f,prediction, "ro-", label="prediction")
plt.plot_date(f,y_test.values, linestyle='dashed',label="actual")
plt.legend(loc="best")

wape=100*np.sum(abs(y_test-prediction))/np.sum(y_test)#лучшая ошибка прогноза
plt.title("Prediction wape error {} %".format(round(wape,2)))
plt.grid(True)
for x,y in zip(f,prediction):

    label = "{:.1f}".format(y)

    ax.annotate(label, 
                 (x,y),
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center') 
    
for x,y in zip(f,y_test.values):

    label = "{:.1f}".format(y)

    ax.annotate(label, 
                 (x,y), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center') 
    

mape=100*np.sum(abs(y_test-prediction)/y_test)/y_test.shape[0]#ошибка прогноза
print("Prediction error mape {} %".format(mape))

print("Prediction error wape {} %".format(wape))


#fact=pd.Series(y_test.values)    
#pr=pd.Series(prediction)

#dat=pd.Series(f.values)
#ns=pd.concat([dat,fact,pr],axis=1)
#ns.columns=['date','fact','pred']
#print(ns)
#ns.to_excel (r'Выгрузка из Питона.xlsx', sheet_name='выгрузка',index = False, header=True)



plt.show()


