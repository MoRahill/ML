import pandas as pd
import yfinance as yf
import math
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = yf.download("GOOGL", start="2000-01-01")
df.columns = df.columns.droplevel(1)

df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']] #These are the features (aka the inputs that are used to train the model)


forecast_col = 'Close'
df.fillna(-99999, inplace=True) 
forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out) #This is the number of days we want to forecast out. We are using 1% of the total length of the dataframe, rounded up to the nearest integer.

df['label'] = df[forecast_col].shift(-forecast_out) #This is the label which is the output that we want to predict. We shift the column up by the number of days we want to forecast out, so that the label for each row is the value of 'Close' that many days in the future.
df.dropna(inplace=True)

X = np.array(df.drop(['label'], axis=1)) #Here 1 represents a column, so we are dropping the 'label' column from the features. We convert the dataframe to a numpy array for use in the machine learning model.
X = preprocessing.scale(X)  #This scales the features so that they have a mean of 0 and a standard deviation of 1. This is important for many machine learning algorithms to perform well.
y = np.array(df['label']) #This is the label column that we want to predict. We convert it to a numpy array as well.

X_lately = X[-forecast_out:]
X = X[:-forecast_out] 
y = y[:-forecast_out]



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2) #This splits the data into a training set and a testing set. The test size is 20% of the total data.
clf = LinearRegression(n_jobs=-1) #This creates a linear regression model.
clf.fit(X_train, y_train) #This trains the model on the training data.
accuracy = clf.score(X_test, y_test) #This evaluates the model on the testing data and returns the coefficient of determination R^2 of the prediction.    

forecast_set = clf.predict(X_lately) #This uses the trained model to make predictions on the data that we want to forecast out.


print(forecast_set, accuracy, forecast_out) #This prints the forecasted values, the accuracy of the model, and the number of days we are forecasting out.

df['Forecast'] = np.nan #This creates a new column in the dataframe called 'Forecast' and fills it with NaN values.
last_date = df.iloc[-1].name #This gets the last date in the dataframe.
last_unix = last_date.timestamp() #This converts the last date to a unix timestamp. 
one_day = 86400
next_unix = last_unix + one_day #This adds one day to the last date to get the next date.

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i] #This adds a new row to the dataframe for each forecasted value, with the date as the index and the forecasted value in the 'Forecast' column.

df['Close'].plot() #This plots the 'Close' column of the dataframe.
df['Forecast'].plot() #This plots the 'Forecast' column of the dataframe.
plt.legend(loc=4) #This adds a legend to the plot in the lower right corner.
plt.xlabel('Date') #This adds a label to the x-axis of the plot.
plt.ylabel('Price') #This adds a label to the y-axis of the plot.
plt.show() #This displays the plot.