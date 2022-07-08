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



def prepareData(data, lag_start, lag_end, test_size):
    
    data.columns = ["y"]

    # считаем индекс в датафрейме, после которого начинается тестовыый отрезок
    
    test_index = int(len(data)*(1-test_size))

    # добавляем лаги исходного ряда в качестве признаков
    for i in range(lag_start, lag_end):
        data["lag_{}".format(i)] = data.y.shift(i)

    #/В зависимости какая у нас временная шкала - или день недели или месяц
    #data["weekday"] = data.index.weekday
    data["weekday"]=data.index.month
   
    # считаем средние только по тренировочной части, чтобы избежать лика
    data['weekday_average'] = data[:test_index].weekday.map(code_mean(data, 'weekday', 'y'))
   
    # выкидываем закодированные средними признаки 
    data.drop(["weekday"], axis=1, inplace=True)
    
    data = data.dropna()
    
    data = data.reset_index(drop=True)
    
    test = int(len(data)*test_size)
    # разбиваем весь датасет на тренировочную и тестовую выборку
    X_train = data.loc[test:].drop(["y"], axis=1)
    y_train = data.loc[test:]["y"]
    X_test = data.loc[:test].drop(["y"], axis=1)
    y_test = data.loc[:test]["y"]

    return X_train, X_test, y_train, y_test

#data = quandl.get("LBMA/GOLD",start_date="2020-01-01",end_date="2020-03-15" ,authtoken="q5TTEoL7sjqJyGu5o1Sz")
#data=data.drop(['USD (PM)','GBP (AM)','GBP (PM)','EURO (AM)' ,'EURO (PM)'],axis=1)


data=pd.read_excel('Продажи в динамике по годам и номенклатуре(01.04.2021).xlsx',\
                   sheet_name='запрос',usecols=('Date','Количество'), parse_dates=['Date'])
data=data.set_index(['Date'],drop=True) 

#data=pd.date_range('2021-04-01',periods=5,freq='M')


#Тестируем
X_train, X_test, y_train, y_test = prepareData(data,lag_start=5, lag_end=11,test_size=0.3)
#lr = MLPRegressor(hidden_layer_sizes=(5,5),activation='relu', solver='lbfgs',learning_rate_init=0.5, max_iter=3000)
#lr = LinearRegression()
lr=RandomForestRegressor()

lr.fit(X_train, y_train)


joblib.dump(lr,'ring.pkl', compress=9)
lr = joblib.load('ring.pkl')#Загрузка модели из файла


prediction = lr.predict(X_test)
#Классный mapping
f=X_test.index.map(lambda x:data.dropna().index[x])


fig,ax=plt.subplots(1,1,figsize=(15, 5))
plt.plot_date(f,prediction, "ro-", label="prediction")
plt.plot_date(f,y_test.values, linestyle='dashed',label="actual")
plt.legend(loc="best")
plt.title("Linear regression\n Mean absolute error {} units".format(round(mean_absolute_error(prediction, y_test))))
plt.grid(True)
for x,y in zip(f,prediction):

    label = "{:.1f}".format(y)

    ax.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    
for x,y in zip(f,y_test.values):

    label = "{:.1f}".format(y)

    ax.annotate(label, # this is the text
                 (x,y), # these are the coordinates to position the label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
    

mape=100*np.sum(abs(y_test-prediction)/y_test)/y_test.shape[0]#ошибка прогноза
print("Ошибка прогноза mape {} %".format(mape))


wape=100*np.sum(abs(y_test-prediction))/np.sum(y_test)#лучшая ошибка прогноза
print("Ошибка прогноза wape {} %".format(wape))

fact=pd.Series(y_test.values)    
pr=pd.Series(prediction)

dat=pd.Series(f.values)
ns=pd.concat([dat,fact,pr],axis=1)
ns.columns=['date','fact','pred']
print(ns)
ns.to_excel (r'f:\Прогнозирование временных рядов (Планирование)\Выгрузка из Питона.xlsx', sheet_name='выгрузка',index = False, header=True)
plt.show()
