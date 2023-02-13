#!/usr/bin/env python
# coding: utf-8

# # Flight Delay Prediction

# In[1]:


#import libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import math
import matplotlib.pyplot as plt

print("Libraries Imported.")


# In[ ]:


# todo
# remove outliers of flight delay
# try PCA to reduce dimensions


# ### Declare parameters for data transformation and modeling

# In[78]:


categorical_cols = ['DAY_OF_WEEK', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'DEST']
continuous_cols = ['DEP_TIME', 'ARR_TIME']
target_col = 'ARR_DELAY'
sample_rate = .25
test_data_frac = .20


# ## Read in BTS Data

# In[61]:


data_raw = pd.read_csv('2022-01-bts-ontime-reporting.csv') #",index_col=0)
print(data_raw.shape)


# ### Remove cancelled and diverted flights

# In[62]:


data_raw = data_raw[data_raw.CANCELLED == 0]
data_raw = data_raw[data_raw.DIVERTED == 0]
print(data_raw.shape)


# ### Histogram of delay arrival

# In[63]:


plt_data = data_raw[data_raw['ARR_DELAY']<240]
plt.hist(plt_data['ARR_DELAY'])
plt.show() 


# ### See unique value count of several input columns

# In[64]:


print(data_raw.DEST_AIRPORT_ID.unique().size)
print(data_raw.DEST_CITY_MARKET_ID.unique().size)
print(data_raw.DEST.unique().size)
print(data_raw.OP_UNIQUE_CARRIER.unique().size)


# ### Sample data to subset rows

# In[65]:


data_sampled = data_raw.sample(frac=sample_rate, random_state=42)
print(data_sampled.shape)


# ### Subset columns

# In[66]:


all_cols = categorical_cols + continuous_cols + [target_col]
data_trimmed = data_sampled[all_cols]
print(data_trimmed.head(5))
print(data_trimmed.shape)


# ### Check for null departure times and arrival times, create new flight time column

# In[67]:


print(data_trimmed['DEP_TIME'].isnull().sum())
print(data_trimmed['ARR_TIME'].isnull().sum())
print(data_trimmed['ARR_DELAY'].isnull().sum())


# In[69]:


#check for flight departing before midnite and landing after midnite
data_trimmed['FLIGHT_TIME'] = data_trimmed['FLIGHT_TIME'].apply(lambda x: x if x >= 0 else 2400 + x)
print(data_trimmed.head(5))
print(data_trimmed.shape)


# ### Round times to hour amounts

# In[49]:


#transform time columns to nearest hour
#data['DEP_TIME'] = round(data['DEP_TIME'] / 100)
#data['ARR_TIME'] = round(data['ARR_TIME'] / 100)
#print(data.head(10))


# ### Show descriptive statistics

# In[80]:


data_trimmed.describe()


# ### Perform one hot encoding on categorical columns

# In[75]:


data_encoded = pd.get_dummies(data_trimmed, columns = categorical_cols)
data_encoded = data_encoded.fillna(0)
print(data_encoded.shape)


# ## Split prepped data into train and test

# In[79]:


from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(data_encoded.drop(target_col,axis=1),data_encoded[target_col],test_size = test_data_frac)

print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)


# ### Linear Regression

# In[89]:


from sklearn.linear_model import LinearRegression

regr = LinearRegression()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_valid)
#y_pred = y_pred.round()
mse = sklearn.metrics.mean_squared_error(y_valid, y_pred)
rmse = math.sqrt(mse)
print("Linear Regression RMSE", rmse)


# In[83]:


y_pred


# In[84]:


y_valid


# ### Decision Tree Regressor

# In[95]:


from sklearn.tree import DecisionTreeRegressor
# Fit regression model

regr = DecisionTreeRegressor(max_depth=5)
regr.fit(x_train, y_train)

# Predict
y_pred_dt = regr.predict(x_valid)
mse = sklearn.metrics.mean_squared_error(y_valid, y_pred_dt)
rmse = math.sqrt(mse)
print("Decision Tree RMSE", rmse)


# ### Random Forest Regression

# In[96]:


from sklearn.ensemble import RandomForestRegressor
  
 # create regressor object
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0, max_features=40)
  
# fit the regressor with x and y data
regressor.fit(x_train, y_train) 
# validate model
y_pred_rf = regressor.predict(x_valid)
mse = sklearn.metrics.mean_squared_error(y_valid, y_pred_rf)
rmse = math.sqrt(mse)
print("Random Forest Regression RMSE", rmse)


# ### Gradient Boosting Regression

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor

reg = GradientBoostingRegressor(random_state=0)
reg.fit(x_train, y_train)

y_pred_gb = reg.predict(x_test)
mse = sklearn.metrics.mean_squared_error(y_valid, y_pred_gb)
rmse = math.sqrt(mse)
print("Gradient Boosting Regression RMSE", rmse)


# ### Neural Network Regression

# In[ ]:


from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

regr = MLPRegressor(random_state=1, max_iter=600).fit(x_train, y_train)
y_pred_nn = regr.predict(x_valid)
mse = sklearn.metrics.mean_squared_error(y_valid, y_pred_nn)
rmse = math.sqrt(mse)
print("Neural Network Regression RMSE", rmse)

