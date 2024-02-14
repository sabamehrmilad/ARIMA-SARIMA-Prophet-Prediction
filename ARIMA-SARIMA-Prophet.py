#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt


# In[2]:


data=pd.read_csv('Urfa_monthly_rainfall_2013_2022.csv', index_col='DATE', parse_dates=True)
data


# In[3]:


data.plot()


# In[4]:


print(data.shape)


# In[5]:


from statsmodels.tsa.stattools import adfuller

result = adfuller(data['PRCP'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])


# In[ ]:





# In[6]:


train = data.iloc[:96]
train


# In[7]:


test=data.iloc[96:]
test


# In[20]:


from pmdarima import auto_arima 
stepwise_fit1 = auto_arima(train['PRCP'], trace=True,seasonal=False, suppress_warnings=True)
stepwise_fit1.summary()


# In[21]:


from pmdarima import auto_arima 
stepwise_fit2 = auto_arima(train['PRCP'], trace=True,seasonal=True, suppress_warnings=True, m=12)
stepwise_fit2.summary()


# In[22]:


arima_fit1 = stepwise_fit1.fit(train['PRCP'])


# In[23]:


sarima_fit2 = stepwise_fit2.fit(train['PRCP'])


# In[29]:


sarima_predictions = sarima_fit2.predict(n_periods=len(test), return_conf_int=False)


# In[24]:


arima_predictions = arima_fit1.predict(n_periods=len(test), return_conf_int=False)


# In[27]:


arima_predictions


# In[30]:


sarima_predictions


# In[31]:


mse_sarima = mean_squared_error(test['PRCP'], sarima_predictions)
rmse_sarima = sqrt(mse_sarima)
mae_sarima = mean_absolute_error(test['PRCP'], sarima_predictions)


# In[32]:


mse_sarima


# In[33]:


rmse_sarima


# In[34]:


mae_sarima


# In[38]:


mse_arima = mean_squared_error(test['PRCP'], arima_predictions)
rmse_arima = sqrt(mse_sarima)
mae_arima = mean_absolute_error(test['PRCP'], arima_predictions)


# In[39]:


mse_arima


# In[40]:


rmse_arima


# In[41]:


mae_arima


# In[10]:


train = train.rename(columns={'DATE': 'ds', 'PRCP': 'y'})


# In[11]:


train


# In[12]:


test = test.rename(columns={'DATE': 'ds', 'PRCP': 'y'})
test


# In[14]:


from prophet import Prophet  


# In[21]:


data2 = data.rename(columns={"PRCP": "y"}).reset_index().rename(columns={"DATE": "ds"})
data2


# In[22]:


# Fit Prophet model
prophet_model = Prophet()
prophet_model.fit(data2)


# # Create a dataframe with future dates for prediction
# future = prophet_model.make_future_dataframe(periods=len(test), freq='M')
# 
# # Make predictions
# prophet_predictions = prophet_model.predict(future)
# 
# # Extract predictions for the test period
# prophet_predictions = prophet_predictions[-len(test):]['yhat']
# 
# # Evaluate performance
# mse_prophet = mean_squared_error(test['value'], prophet_predictions)
# rmse_prophet = sqrt(mse_prophet)
# mae_prophet = mean_absolute_error(test['value'], prophet_predictions)

# In[23]:


future = prophet_model.make_future_dataframe(periods=len(test), freq='M')


# In[24]:


prophet_predictions = prophet_model.predict(future)


# In[25]:


prophet_predictions = prophet_predictions[-len(test):]['yhat']


# In[28]:


mse_prophet = mean_squared_error(test['y'], prophet_predictions)
rmse_prophet = sqrt(mse_prophet)
mae_prophet = mean_absolute_error(test['y'], prophet_predictions)


# In[29]:


mse_prophet 


# In[30]:


rmse_prophet 


# In[31]:


mae_prophet 


# # TAVERAGE

# In[45]:


data2=pd.read_csv('Urfa_monthly_Taverage_2013_2023 (3).csv' , index_col='Month')
data2


# In[48]:


data3 = data2.rename(columns={"TAVG": "y"}).reset_index().rename(columns={"Month": "ds"})
data3


# In[51]:


data4 = data3.set_index('ds')
data4


# In[52]:


data4.plot()


# In[53]:


print(data4.shape)


# In[54]:


data4


# In[56]:


data4 = data4.loc[~((data4.index == '2023-01-01') & (data4.index <= '2023-12-31'))]
data4


# In[57]:


data4 = data4[data4.index != '2023-01']


# In[58]:


data4


# In[59]:


data4.plot()


# In[60]:


print(data4.shape)


# In[61]:


from statsmodels.tsa.stattools import adfuller

result = adfuller(data['y'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])


# In[62]:


train = data4.iloc[:96]
train


# In[63]:


test = data4.iloc[96:]
test 


# In[64]:


from pmdarima import auto_arima 
stepwise_fit3 = auto_arima(train['y'], trace=True,seasonal=False, suppress_warnings=True)
stepwise_fit3.summary()


# In[65]:


from pmdarima import auto_arima 
stepwise_fit4 = auto_arima(train['y'], trace=True,seasonal=True, suppress_warnings=True, m=12)
stepwise_fit4.summary()


# In[70]:


#Arima
arima_fit3 = stepwise_fit3.fit(train['y'])


# In[71]:


sarima_fit4 = stepwise_fit4.fit(train['y'])


# In[72]:


arima_predictions_T = arima_fit3.predict(n_periods=len(test), return_conf_int=False)


# In[73]:


sarima_predictions_t = sarima_fit4.predict(n_periods=len(test), return_conf_int=False)


# In[74]:


arima_predictions_T


# In[75]:


sarima_predictions_t


# In[87]:


mse_arima_T = mean_squared_error(test['y'], arima_predictions_T)
rmse_arima_T = sqrt(mse_sarima)
mae_arima_T = mean_absolute_error(test['y'], arima_predictions_T)


# In[88]:


mse_arima_T


# In[89]:


rmse_arima_T


# In[90]:


mae_arima_T 


# In[91]:


mse_sarima = mean_squared_error(test['y'], sarima_predictions_t)
rmse_sarima = sqrt(mse_sarima)
mae_sarima = mean_absolute_error(test['y'], sarima_predictions_t)


# In[92]:


mse_sarima


# In[93]:


rmse_sarima


# In[94]:


mae_sarima


# In[96]:


data3


# In[99]:


data3 = data3[(data3['ds'] <= '2023-01-01') | (data3['ds'] > '2023-12-31')]
data3


# In[101]:


data3 = data3[data3['ds'] != '2023-01']
data3


# In[102]:


prophet_model = Prophet()
prophet_model.fit(data3)


# In[103]:


future2 = prophet_model.make_future_dataframe(periods=len(test), freq='M')


# In[104]:


prophet_predictions2 = prophet_model.predict(future2)


# In[106]:


prophet_predictions2 = prophet_predictions2[-len(test):]['yhat']


# In[107]:


mse_prophet = mean_squared_error(test['y'], prophet_predictions2)
rmse_prophet = sqrt(mse_prophet)
mae_prophet = mean_absolute_error(test['y'], prophet_predictions2)


# In[108]:


mse_prophet 


# In[109]:


mae_prophet 


# In[110]:


rmse_prophet 


# In[ ]:




