import logging
logging.captureWarnings(True)
import openpyxl
import pandas as pd
import datetime as dt
import numpy as np
from datetime import timedelta, date
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import quandl
from sklearn.metrics import mean_absolute_error
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import joblib



def code_mean(data, cat_feature, real_feature):
    """
    Возвращает словарь, где ключами являются уникальные категории признака cat_feature, 
    а значениями - средние по real_feature
    """
    return dict(data.groupby(cat_feature)[real_feature].mean())



def prepareData(data, lag_start, lag_end):
    
    data.columns = ["y"]

    # считаем индекс в датафрейме, после которого начинается тестовыый отрезок
    
   

    # добавляем лаги исходного ряда в качестве признаков
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.y.shift(i)

    #/В зависимости какая у нас временная шкала - или день недели или месяц
    #data["weekday"] = data.index.weekday
    data["weekday"]=data.index.month
   
    # считаем средние только по тренировочной части, чтобы избежать лика
    data['weekday_average'] = data.weekday.map(code_mean(data, 'weekday', 'y'))
   
    # выкидываем закодированные средними признаки 
    data.drop(["weekday"], axis=1, inplace=True)
    
    #data = data.reset_index(drop=True)
    X_test = data[-lag_start:].drop(["y"], axis=1)
    X_test = X_test.dropna()
    return X_test
    
   

data=pd.read_excel('Продажи в динамике по годам и номенклатуре(01.04.2021).xlsx',\
                   sheet_name='запрос',usecols=('Date','Количество'), parse_dates=['Date'])
data=data.set_index(['Date'],drop=True) 

#data=pd.date_range('2021-04-01',periods=5,freq='M')


#Тестируем
X_test = prepareData(data,lag_start=5, lag_end=9)


case = joblib.load('ring.pkl')#Загрузка модели из файла


prediction = case.predict(X_test)
#Классный mapping
#f=X_test.index.map(lambda x:data.dropna().index[x])

fig,ax=plt.subplots(1,1,figsize=(15, 5))
plt.plot_date(X_test.index,prediction, "ro-", label="prediction")
plt.legend(loc="best")

plt.grid(True)
for x,y in zip(X_test.index,prediction):

    label = "{:.1f}".format(y)

    ax.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment cfor x,y in zip(f,prediction):

   

  
plt.show()



