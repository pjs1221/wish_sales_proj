# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:13:42 2020

@author: shinp
"""

from datetime import datetime
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(),3600)
        tmin, tsec = divmod(temp_sec,60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour,tmin,round(tsec,2)))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Data/data_eda.csv')
tags = pd.read_csv('Data/unique-categories.sorted-by-count.csv')

#Choose relevant columns
df.columns

df_model = df[['units_sold','price','retail_price','rating','badge_local_product',
               'badge_product_quality','product_color','product_variation_size_id',
               'shipping_is_express','origin_country','has_urgency_banner','merchant_rating',
               'num_listings','image_contains_person']]

#One-hot encoding for tag variable
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()

ohe_tags = pd.DataFrame(mlb.fit_transform(df['tags'].dropna().str.split(',')),columns=mlb.classes_, index=df.index)
used_tags = pd.DataFrame()
for i in range(0,50):
    temp_tag = tags['keyword'][i]
    if(temp_tag in ohe_tags.columns):
        used_tags[temp_tag] = ohe_tags[temp_tag]
        
    

#Categorize variables
def categorize(units):
    if(units > 0 and units <= 100):
        return '0 to 100 units'
    elif(units > 100 and units <= 1000):
        return '1000 units'
    elif(units > 1000 and units <= 5000):
        return '5000 units'
    elif(units > 5000 and units <= 10000):
        return '10000 units'
    elif(units > 10000 and units <= 20000):
        return '20000 units'
    elif(units > 20000):
        return 'Greater than 20000 units'
    
df_model['units_sold'] = df_model['units_sold'].apply(lambda x: categorize(x))

#Concatenate models
include_tags = True
if(include_tags):
    df_model = pd.concat([df_model, used_tags], axis=1)


#train test split
from sklearn.model_selection import train_test_split

X = df_model.drop('units_sold',axis=1)
y = df_model.units_sold.values


 #Get dummies
X = pd.get_dummies(X)

 #Label encode y values
from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

"""
# multiple linear regression
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train,y_train)

np.mean(cro)
"""
from sklearn.model_selection import cross_val_score


#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

scores = []
for i in range(1,25):
    print(i)
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)

    knn_scores = cross_val_score(knn,X_train, y_train, cv =10)
    knn_avg = np.mean(knn_scores)
    scores.append(knn_avg)
    
plt.plot(range(1,25),scores)
plt.show()
    

# SVM with linear kernel
from sklearn import svm

svm_clf = svm.SVC()
svm_clf.fit(X_train,y_train)

svm_scores = cross_val_score(svm_clf,X_train, y_train, cv =10)
svm_avg = np.mean(svm_scores)

print('Cross Validation Accuracy Scores:',svm_scores)
print('Cross Validation Accuracy Mean:',svm_avg)


# Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train,y_train)

dtree_scores = cross_val_score(dtree,X_train, y_train, cv =10)
dtree_avg = np.mean(dtree_scores)

print('Cross Validation Accuracy Scores:',dtree_scores)
print('Cross Validation Accuracy Mean:',dtree_avg)

# Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf = rf.fit(X_train,y_train)

rf_scores = cross_val_score(rf,X_train, y_train, cv =10)
rf_avg = np.mean(rf_scores)

print('Cross Validation Accuracy Scores:',rf_scores)
print('Cross Validation Accuracy Mean:',rf_avg)

# XGBoost

import xgboost
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train,y_train)

xgb_scores = cross_val_score(xgb,X_train,y_train, cv=10)
xgb_avg = np.mean(xgb_scores)

print('Cross Validation Accuracy Scores:',xgb_scores)
print('Cross Validation Accuracy Mean:',xgb_avg)


# tune models GridsearchCV

from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 2)

start_time = timer(None)
grid_search.fit(X_train,y_train)
timer(start_time)

grid_search.best_params_
best_grid = grid_search.best_estimator_

# test ensembles

from sklearn import metrics
tpred_dtree = dtree.predict(X_test)
tpred_rf = best_grid.predict(X_test)
tpred_xgb = xgb.predict(X_test)

dtree_accuracy = metrics.accuracy_score(y_test,tpred_dtree)
rf_accuracy = metrics.accuracy_score(y_test,tpred_rf)
xgb_accuracy = metrics.accuracy_score(y_test,tpred_xgb)

print('Dtree Accuracy',)
print('Dtree Accuracy',)