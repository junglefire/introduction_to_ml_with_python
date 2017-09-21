from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

svm = SVC(C=100) 
svm.fit(X_train, y_train) 
print("(original dataset) Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))

# preprocessing using 0-1 scaling 
scaler = MinMaxScaler() 
scaler.fit(X_train) 
X_train_scaled = scaler.transform(X_train) 
X_test_scaled = scaler.transform(X_test)

# learning an SVM on the scaled training data 
svm.fit(X_train_scaled, y_train)

# scoring on the scaled test set 
print("(min-max scaled dataset) Test set accuracy: {:.2f}".format( svm.score(X_test_scaled, y_test)))

# preprocessing using zero mean and unit variance scaling 
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() 
scaler.fit(X_train) 
X_train_scaled = scaler.transform(X_train) 
X_test_scaled = scaler.transform(X_test)

# learning an SVM on the scaled training data 
svm.fit(X_train_scaled, y_train)

# scoring on the scaled test set 
print("(standard scaled dataset) Test set accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))



