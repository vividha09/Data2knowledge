import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error,mean_squared_error

closing_stock = pd.read_csv('BAJFINANCE.ns-4.csv',index_col='Date', parse_dates=True)
forecast_data = pd.read_csv('BAJFINANCE.ns-4.csv',index_col='Date', parse_dates=True)

train_stock = forecast_data[:65]
test_stock = forecast_data[65:]


"""decompose_result = seasonal_decompose(closing_stock["Close"],model="multiplicative")
decompose_result.plot();
"""

closing_stock.index.freq = "MS"
forecast_data.index.freq = "MS"
m = 12
alpha = 1/(2*m)

"""closing_stock["HWES3_ADD"] = ExponentialSmoothing(closing_stock["Close"],trend="add",seasonal="add",seasonal_periods=12).fit().fittedvalues
closing_stock["HWES3_MUL"] = ExponentialSmoothing(closing_stock["Close"],trend="mul",seasonal="mul",seasonal_periods=12).fit().fittedvalues
closing_stock[["Close","HWES3_ADD","HWES3_MUL"]].plot(title="Holt Winters Triple Exponential Smoothing: Additive and Multiplicative Seasonality");
"""

fitted_model = ExponentialSmoothing(train_stock['Close'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
test_predictions = fitted_model.forecast(26)
train_stock['Close'].plot(legend=True,label='TRAIN')
test_stock['Close'].plot(legend=True,label='TEST',figsize=(6,4))
test_predictions.plot(legend=True,label='PREDICTIONS')
plt.title('Train, Test and Predicted Test using Holt Winters')


test_stock['Close'].plot(legend=True,label='TEST',figsize=(9,6))
test_predictions.plot(legend=True,label='PREDICTION');
