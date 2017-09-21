from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.model_selection import LeaveOneOut

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

iris = load_iris() 
logreg = LogisticRegression()

loo = LeaveOneOut() 
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("Number of cv iterations: ", len(scores))
print("Mean accuracy: {:.2f}".format(scores.mean()))
