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

from sklearn.metrics import mean_absolute_error
from pandas.plotting import register_matplotlib_converters
import sklearn.model_selection as model_selection
import joblib


df=pd.read_excel('f:\gorserv\Gorserv.xlsx',sheet_name='Лист1',usecols=("A:Q"))


#Тестируем
X_train, X_test, y_train, y_test = model_selection.train_test_split(df.iloc[:,:df.shape[1]-1].values,\
                                                                    df.iloc[:,[df.shape[1]-1]].values, train_size=0.70,test_size=0.30, random_state=7)
#lr = MLPRegressor(hidden_layer_sizes=(5,5),activation='relu', solver='lbfgs',learning_rate_init=0.5, max_iter=3000)
lr = LinearRegression()
#lr=RandomForestRegressor()

lr.fit(X_train, y_train)



prediction = lr.predict(X_test)


fig,ax=plt.subplots(1,1,figsize=(15, 5))
plt.plot(prediction[200:250], "ro-", label="prediction")
plt.plot(y_test[300:350].ravel(), linestyle='dashed',label="actual")
plt.legend(loc="best")
plt.title("Linear regression\n Mean absolute error {} units".format(round(mean_absolute_error(prediction, y_test))))
mape=100*np.sum(abs(y_test-prediction)/y_test)/y_test.shape[0]#ошибка прогноза
print("Ошибка прогноза mape {} %".format(mape))


wape=100*np.sum(abs(y_test-prediction))/np.sum(y_test)
print("Ошибка прогноза wape {} %".format(wape))
plt.show()
