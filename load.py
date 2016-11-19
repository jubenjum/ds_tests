#!/usr/bin/python -W ignore::DeprecationWarning

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 19:16:05 2016

@author: juan
"""


from xgboost import XGBRegressor
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import multiprocessing


dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

def report(results, n_top=1000):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            msg = "Model with rank: {} Mean validation score: {:.3f} (std: {:.3f}) Parameters: {} ".format(i, results['mean_test_score'][candidate], results['std_test_score'][candidate], results['params'][candidate])

            print(msg)

#%%  public data

# 
data_train = pd.read_csv("data/public_train.csv", 
                         parse_dates=['DateOfDeparture'], date_parser=dateparse)

data_ = data_train

##%% competitors data
data_competitors = pd.read_csv("data/Competitors1.csv", 
                               names=["Id", "Departure", "Arrival", "Airline", 
                               "Competitors", "Codeshare"], 
                               skiprows=1)
                                      
data_competitors = data_competitors.drop([
    "Id",
    #"Departure",
    #"Arrival",
    "Airline",
    #"Competitors",
    #"Codeshare"
    ], axis=1)

data_ = pd.merge(data_, data_competitors, 
                 how='left', 
                 left_on=['Departure', 'Arrival'], 
                 right_on=['Departure','Arrival'])

#%% holidays
# Date,National Holiday,Special Day
data_holidays = pd.read_csv("data/national_holidays.csv", 
                            parse_dates=['Date'], date_parser=dateparse)
data_holidays = data_holidays.drop([
    #'Date',
    'National Holiday', 
    'Special Day'
    ], axis=1)
data_holidays = data_holidays.rename(columns={'Date': 'holiday'}) 
data_ = pd.merge(data_, data_holidays, how='left', 
                 left_on=['DateOfDeparture'], 
                 right_on=['holiday'])
data_['holiday'] = pd.to_datetime(data_['holiday']).dt.week.fillna(0)
data_.ix[data_.holiday > 0, 'holiday'] = 1.0

#%% weather
# Date, AirPort, Max TemperatureC,Mean TemperatureC,Min TemperatureC,Dew PointC,MeanDew PointC,Min DewpointC,Max Humidity,Mean Humidity,Min Humidity,Max Sea Level PressurehPa,Mean Sea Level PressurehPa,Min Sea Level PressurehPa,Max VisibilityKm,Mean VisibilityKm,Min VisibilitykM,Max Wind SpeedKm/h,Mean Wind SpeedKm/h,Max Gust SpeedKm/h,Precipitationmm,CloudCover,Events,WindDirDegrees

data_weather = pd.read_csv("data/weather.csv", 
                            parse_dates=['Date'], date_parser=dateparse)
data_weather = data_weather.drop([
    #'Date', 
    #'AirPort', 
    #'Max TemperatureC',
    'Mean TemperatureC', 
    'Min TemperatureC', 
    'Dew PointC',
    'MeanDew PointC','Min DewpointC',
    'Max Humidity', 
    'Mean Humidity',
    'Min Humidity',
    'Max Sea Level PressurehPa',
    'Mean Sea Level PressurehPa',
    'Min Sea Level PressurehPa', 
    'Max VisibilityKm', 
    'Mean VisibilityKm',
    'Min VisibilitykM',
    'Max Wind SpeedKm/h',
    'Mean Wind SpeedKm/h',
    'Max Gust SpeedKm/h',
    'Precipitationmm',
    'CloudCover',
    'Events',
    'WindDirDegrees'
    ], axis=1)

data_ = pd.merge(data_, data_weather, 
                 how='left', 
                 left_on=['DateOfDeparture', 'Departure'], 
                 right_on=['Date', 'AirPort'])
data_ = data_.drop(['Date', 'AirPort'], axis=1)
data_ =  data_.rename(columns={'Max TemperatureC': 'max_temp_Departure'})

data_ = pd.merge(data_, data_weather, 
                 how='left', 
                 left_on=['DateOfDeparture', 'Arrival'], 
                 right_on=['Date', 'AirPort'])
data_ = data_.drop(['Date', 'AirPort'], axis=1)
data_ = data_.rename(columns={'Max TemperatureC': 'max_temp_Arrival'})

#%% price jet fuel
#Date,price
dateparse = lambda x: pd.datetime.strptime(x, '%d-%m-%Y')
data_jet_fuel = pd.read_csv("data/price_jet_fuel.csv", 
                            parse_dates=['Date'], date_parser=dateparse)

data_ = pd.merge(data_, data_jet_fuel, 
                 how='left', 
                 left_on=['DateOfDeparture'], 
                 right_on=['Date'])

data_ = data_.drop(['Date'], axis=1)


#%% Preprocessing for prediction
data_encoded = data_
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Departure'], prefix='d'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['Arrival'], prefix='a'))
data_encoded = data_encoded.drop('Departure', axis=1)
data_encoded = data_encoded.drop('Arrival', axis=1)

# following http://stackoverflow.com/questions/16453644/regression-with-date-variable-using-scikit-learn
data_encoded['DateOfDeparture'] = pd.to_datetime(data_encoded['DateOfDeparture'])
data_encoded['year'] = data_encoded['DateOfDeparture'].dt.year
data_encoded['month'] = data_encoded['DateOfDeparture'].dt.month
data_encoded['day'] = data_encoded['DateOfDeparture'].dt.day
data_encoded['weekday'] = data_encoded['DateOfDeparture'].dt.weekday
data_encoded['week'] = data_encoded['DateOfDeparture'].dt.week
data_encoded['n_days'] = data_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

data_encoded = data_encoded.join(pd.get_dummies(data_encoded['year'], prefix='y'))
#data_encoded = data_encoded.join(pd.get_dummies(data_encoded['month'], prefix='m'))
#data_encoded = data_encoded.join(pd.get_dummies(data_encoded['day'], prefix='d'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['weekday'], prefix='wd'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['week'], prefix='w'))

#  A linear regressor baseline
drops = ['DateOfDeparture', 'log_PAX' ]#, 'month', 'day', 'weekday', 'week', 'n_days' ]

features = data_encoded.drop(drops, axis=1)
X_columns = data_encoded.columns.drop(drops)
X = features.values
y = data_encoded['log_PAX'].values

#%% baseline linear regression
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

#%%

#####initial run
#n_estimators = 85
#max_depth = 65
#max_features = 65
#reg = RandomForestRegressor(n_estimators=n_estimators, 
#                            max_depth=max_depth, 
#                            max_features=max_features)
#
#scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error',n_jobs=3)
#print("log RMSE: {:.4f} +/-{:.4f}".format(np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores))))
#aram_grid = {"n_estimators": range(20, 100, 40),
#       "max_depth": range(20, 100, 40),
#       "max_features": range(20, 100, 40)} 
#
#reg.fit(X_train, y_train)
#plt.figure(figsize=(15, 5))
#ordering = np.argsort(reg.feature_importances_)[::-1][:50]
#importances = reg.feature_importances_[ordering]
#feature_names = X_columns[ordering]
#x = np.arange(len(feature_names))
#plt.bar(x, importances)
#plt.xticks(x + 0.5, feature_names, rotation=90, fontsize=15);
#plt.show()
#plt.savefig('test.png')


###if multiprocessing.cpu_count() < 40:
###    #scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error',n_jobs=3)
###    #print("log RMSE: {:.4f} +/-{:.4f}".format(np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores))))
###    param_grid = {"n_estimators": range(20, 100, 40),
###            "max_depth": range(20, 100, 40),
###            "max_features": range(20, 100, 40)} 
###
###    # run grid search                                           
###    grid_search = GridSearchCV(reg, param_grid=param_grid, n_jobs=2, verbose=1)      
###    start = time()                                              
###    grid_search.fit(X_train, y_train)                                       
###    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
###          % (time() - start, len(grid_search.cv_results_['params'])))
###    report(grid_search.cv_results_)
###
###
###else:
###    # grid search parameters
###    param_grid = {"n_estimators": range(20,100, 5),                       
###                  "max_depth": range(20, 100, 5),                   
###                  "max_features": range(20, 100, 5)}             
###
###    # run grid search                                           
###    grid_search = GridSearchCV(reg, param_grid=param_grid, n_jobs=40, verbose=1)      
###    start = time()                                              
###    grid_search.fit(X_train, y_train)                                       
###    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
###          % (time() - start, len(grid_search.cv_results_['params'])))
###    report(grid_search.cv_results_)


##### xgboost
max_depth=50
learning_rate=0.2
n_estimators=60

reg = XGBRegressor(max_depth=max_depth, 
        learning_rate=learning_rate, 
        n_estimators=n_estimators,
        nthread=1)

# grid search parameters
param_grid = {"max_depth": range(20,100, 10),                       
              "n_estimators": range(20, 100, 10),                   
              "learning_rate": [0.1, 0.2, 0.3, 0.4]}             

# run grid search                                           
grid_search = GridSearchCV(reg, param_grid=param_grid, n_jobs=2, verbose=1)      
start = time()                                              
grid_search.fit(X_train, y_train)                                       
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)



#scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='mean_squared_error')  
#print("log RMSE: {:.4f} +/-{:.4f}".format(                                           
#    np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores))))                            
                                                                                      
                                                                                      












