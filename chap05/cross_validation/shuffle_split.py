from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.model_selection import ShuffleSplit

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

iris = load_iris() 
logreg = LogisticRegression()

shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("Cross-validation scores:\n{}".format(scores))
