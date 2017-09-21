from sklearn.datasets import make_blobs 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

# create a synthetic dataset 
X, y = make_blobs(random_state=0) 

# split data and labels into a training and a test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
# instantiate a model and fit it to the training set 
logreg = LogisticRegression().fit(X_train, y_train) 
# evaluate the model on the test set 
print("Test set score: {:.2f}".format(logreg.score(X_test, y_test)))



