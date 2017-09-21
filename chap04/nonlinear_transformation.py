from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)

print("Number of feature appearances:\n{}".format(np.bincount(X[:, 0])))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
score = Ridge().fit(X_train, y_train).score(X_test, y_test) 
print("Test score: {:.3f}".format(score))

X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)
score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test) 
print("Test score: {:.3f}".format(score))

plt.hist(np.log(X_train_log[:, 0] + 1), bins=25, color='b') 
plt.ylabel("Number of appearances")
plt.xlabel("Value")
plt.show()