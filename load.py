# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 19:16:05 2016

@author: juan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor


dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

#%%  public data

# 
data_train = pd.read_csv("data/public_train.csv", 
                         parse_dates=['DateOfDeparture'], date_parser=dateparse)

#%% competitors data
data_competitors = pd.read_csv("data/Competitors1.csv", 
                               names=["Id", "Departure", "Arrival", "Airline", 
                               "Competitors", "Codeshare"], 
                               skiprows=1)
                                      
data_competitors = data_competitors.drop([
    "Id",
    #"Departure",
    #"Arrival",
    "Airline",
    "Competitors",
    "Codeshare"
    ], axis=1)

data_ = pd.merge(data_train, data_competitors, 
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

# TODO Correct name of temperatures
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

#%% runways
#"id","airport_ref","airport_ident","length_ft","width_ft","surface","lighted","closed","le_ident","le_latitude_deg","le_longitude_deg","le_elevation_ft","le_heading_degT","le_displaced_threshold_ft","he_ident","he_latitude_deg","he_longitude_deg","he_elevation_ft","he_heading_degT","he_displaced_threshold_ft" 
#data_runway = pd.read_csv("data/runway.csv")
#data_runway = data_runway.drop([
#    "id",
#    #"airport_ref",
#    "airport_ident",
#    "length_ft",
#    "width_ft",
#    #"surface",
#    "lighted",
#    "closed",
#    "le_ident",
#    "le_latitude_deg",
#    "le_longitude_deg",
#    "le_elevation_ft",
#    "le_heading_degT",
#    "le_displaced_threshold_ft",
#    "he_ident",
#    "he_latitude_deg",
#    "he_longitude_deg",
#    "he_elevation_ft",
#    "he_heading_degT",
#    "he_displaced_threshold_ft" 
#    ], axis=1)
#
#data_ = pd.merge(data_, data_runway, 
#                 how='left', 
#                 left_on=['DateOfDeparture', 'Departure'], 
#                 right_on=['Date', 'AirPort'])

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
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['month'], prefix='m'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['day'], prefix='d'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['weekday'], prefix='wd'))
data_encoded = data_encoded.join(pd.get_dummies(data_encoded['week'], prefix='w'))

#  A linear regressor baseline

features = data_encoded.drop(['log_PAX','DateOfDeparture'], axis=1)
X_columns = data_encoded.columns.drop(['log_PAX','DateOfDeparture'])
X = features.values
y = data_encoded['log_PAX'].values

#%% baseline linear regression
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

## linear
#reg = LinearRegression()
#
#scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='mean_squared_error')
#print("log RMSE: {:.4f} +/-{:.4f}".format(
#    np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores))))

#%%
n_estimators = 40
max_depth = 60
max_features = 60

reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features)
scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='mean_squared_error',n_jobs=3)
print("log RMSE: {:.4f} +/-{:.4f}".format(
    np.mean(np.sqrt(-scores)), np.std(np.sqrt(-scores))))
    
#reg.fit(X_train, y_train)
#plt.figure(figsize=(15, 5))
#ordering = np.argsort(reg.feature_importances_)[::-1][:50]
#importances = reg.feature_importances_[ordering]
#feature_names = X_columns[ordering]
#x = np.arange(len(feature_names))
#plt.bar(x, importances)
#plt.xticks(x + 0.5, feature_names, rotation=90, fontsize=15);

##%% adaboost
#max_depth = 5 # tree depth
#n_trees = 200 # number of trees
#max_features = None # number of random features at each cut
#n_samples = X_train.shape[0]
#w = np.ones(n_samples) / n_samples
#
#training_errors = []
#test_errors = []
#models = []
#alphas = []
#training_errors_h = []
#ts = plt.arange(n_trees)
#
#for t in range(n_trees):
#    # Your code should go here
#    clf = DecisionTreeClassifier(
#        max_depth=max_depth, max_features=max_features)
#    clf.fit(X_train, y_train, sample_weight=w)
#    y_pred = clf.predict(X_train)
#    training_error_h = np.mean(y_pred != y_train)
#    training_errors_h.append(training_error_h)
#    gamma = np.sum(w * y_train * y_pred)
#    #print gamma
#    alpha = 0.5 * np.log((1. + gamma) / (1. - gamma))
#    alphas.append(alpha)
#    models.append(clf)
#    
#    # udpate weights
#    good = (y_pred == y_train)
#    w[good] *= 1. / (1. + gamma)
#    w[~good] *= 1. / (1. - gamma)
#    
#    y_disc_train_global = sum(
#        alpha * clf.predict(X_train) for alpha, clf in zip(alphas, models))
#    y_disc_test_global = sum(
#        alpha * clf.predict(X_test) for alpha, clf in zip(alphas, models))
#    
#    y_pred_train_global = np.sign(y_disc_train_global)
#    y_pred_test_global = np.sign(y_disc_test_global)
#    
#    training_error = np.mean(y_pred_train_global != y_train)
#    test_error = np.mean(y_pred_test_global != y_test)
#
#    training_errors.append(training_error)
#    test_errors.append(test_error)
#    
#    plt.clf()
#    plt.plot(ts[:t+1], training_errors[:t+1], c='b')
#    plt.plot(ts[:t+1], test_errors[:t+1], c='r')
#    display.clear_output(wait=True)
#    display.display(plt.gcf())
#    sleep(.001)



