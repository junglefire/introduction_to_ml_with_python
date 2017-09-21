from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("(ridge alpha = 0.10) Training set score: {:.2f}".format(ridge01.score(X_train, y_train))) 
print("(ridge alpha = 0.10) Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

lasso = Lasso().fit(X_train, y_train)
print("(lasso alpha = 1) Training set score: {:.2f}".format(lasso.score(X_train, y_train))) 
print("(lasso alpha = 1) Test set score: {:.2f}".format(lasso.score(X_test, y_test))) 
print("(lasso alpha = 1) Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

# we increase the default setting of "max_iter",
# otherwise the model would warn us that we should increase max_iter. 
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train) 
print("(lasso alpha = 0.01) Training set score: {:.2f}".format(lasso001.score(X_train, y_train))) 
print("(lasso alpha = 0.01) Test set score: {:.2f}".format(lasso001.score(X_test, y_test))) 
print("(lasso alpha = 0.01) Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

# if we set alpha too low, however, we again remove the effect of 
# regularization and end up overfitting, with a result similar to 
# LinearRegression
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train) 
print("(lasso alpha = 0.0001) Training set score: {:.2f}".format(lasso00001.score(X_train, y_train))) 
print("(lasso alpha = 0.0001) Test set score: {:.2f}".format(lasso00001.score(X_test, y_test))) 
print("(lasso alpha = 0.0001) Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()
