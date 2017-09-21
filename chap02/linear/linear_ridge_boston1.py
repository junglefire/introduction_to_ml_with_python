from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression().fit(X_train, y_train)
print("(lr) Training set score: {:.2f}".format(lr.score(X_train, y_train))) 
print("(lr) Test set score: {:.2f}".format(lr.score(X_test, y_test)))

ridge = Ridge().fit(X_train, y_train)
print("(alpha = 1.00) Training set score: {:.2f}".format(ridge.score(X_train, y_train))) 
print("(alpha = 1.00) Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("(alpha = 10.0) Training set score: {:.2f}".format(ridge10.score(X_train, y_train))) 
print("(alpha = 10.0) Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("(alpha = 0.10) Training set score: {:.2f}".format(ridge01.score(X_train, y_train))) 
print("(alpha = 0.10) Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()