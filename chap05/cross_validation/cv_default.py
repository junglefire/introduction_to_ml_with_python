from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

iris = load_iris() 
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target) 
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))

scores = cross_val_score(logreg, iris.data, iris.target, cv=5) 
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))


