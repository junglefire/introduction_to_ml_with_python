from sklearn.datasets import make_blobs 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

iris = load_iris() 
logreg = LogisticRegression()

kfold = KFold(n_splits=5) 
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold) 
print("Cross-validation scores:\n{}".format(scores))

kfold = KFold(n_splits=3) 
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print("Cross-validation scores:\n{}".format(scores))

kfold = KFold(n_splits=3, shuffle=True, random_state=0)
scores = cross_val_score(logreg, iris.data, iris.target, cv=kfold)
print("Cross-validation scores:\n{}".format(scores))
