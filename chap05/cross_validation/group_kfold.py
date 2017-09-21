from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.model_selection import GroupKFold

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

logreg = LogisticRegression()

# create synthetic dataset 
X, y = make_blobs(n_samples=12, random_state=0)
# assume the first three samples belong to the same group, 
# then the next four, etc.
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
print("Cross-validation scores:\n{}".format(scores))