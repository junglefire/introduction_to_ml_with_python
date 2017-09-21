from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

X, y = mglearn.datasets.make_wave(n_samples=100) 
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

bins = np.linspace(-3, 3, 11) 
print("bins: {}".format(bins))

which_bin = np.digitize(X, bins=bins) 
print("\nData points:\n", X[:5]) 
print("\nBin membership for data points:\n", which_bin[:5])
print("\n")

# transform using the OneHotEncoder 
encoder = OneHotEncoder(sparse=False) 

# encoder.fit finds the unique values that appear in which_bin 
encoder.fit(which_bin) 
# transform creates the one-hot encoding 
X_binned = encoder.transform(which_bin) 
print(X_binned[:5])
print("X_binned.shape: {}".format(X_binned.shape))

X_product = np.hstack([X_binned, X * X_binned]) 
print(X_product.shape)

line_binned = encoder.transform(np.digitize(line, bins=bins))
line_combined = np.hstack([line, line_binned]) 

reg = LinearRegression().fit(X_product, y)

line_product = np.hstack([line_binned, line * line_binned]) 
plt.plot(line, reg.predict(line_product), label='linear regression product')

for bin in bins:
    plt.plot([bin, bin], [-3, 3], ':', c='k')
    plt.plot(X[:, 0], y, 'o', c='k')
    plt.ylabel("Regression output")
    plt.xlabel("Input feature")
    plt.legend(loc="best")

plt.show()
