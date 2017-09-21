from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import *

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

citibike = mglearn.datasets.load_citibike()
print("Citi Bike data:\n{}".format(citibike.head()))

# extract the target values (number of rentals) 
y = citibike.values 
# convert the time to POSIX time using "%s" 
X = citibike.index.strftime("%s").astype("int").reshape(-1, 1)

# use the first 184 data points for training, and the rest for testing 
n_train = 184

# function to evaluate and plot a regressor on a given feature set 
def eval_on_features(features, target, regressor):
    # split the given features into a training and a test set 
    X_train, X_test = features[:n_train], features[n_train:] 
    # also split the target array 
    y_train, y_test = target[:n_train], target[n_train:] 
    regressor.fit(X_train, y_train) 
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test, y_test))) 
    y_pred = regressor.predict(X_test) 
    y_pred_train = regressor.predict(X_train) 
    plt.figure(figsize=(10, 3))
    plt.xticks(range(0, len(X), 8), strftime("%a %m-%d"), rotation=90, ha="left")
    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")
    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label="prediction test")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("Date")
    plt.ylabel("Rentals")

X_hour_week = np.hstack([citibike.index.dayofweek.astype("int64").values.reshape(-1, 1), citibike.index.hour.astype("int64").values.reshape(-1, 1)]) 

enc = OneHotEncoder() 
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()

poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge() 
eval_on_features(X_hour_week_onehot_poly, y, lr)

plt.show()
