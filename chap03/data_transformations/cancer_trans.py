from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

# the dataset contains 569 data points, each represented by 30 measurements
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1) 

print(X_train.shape) 
print(X_test.shape)

scaler = MinMaxScaler()
scaler.fit(X_train)
print(scaler)

# transform data 
X_train_scaled = scaler.transform(X_train) 

# print dataset properties before and after scaling 
print("transformed shape: {}".format(X_train_scaled.shape)) 
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0))) 
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0))) 
print("per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0))) 
print("per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))

# transform test data 
X_test_scaled = scaler.transform(X_test) 

# print test data properties after scaling 
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0))) 
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))



