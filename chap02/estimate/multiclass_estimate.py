from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

iris = load_iris() 
X_train, X_test, y_train, y_test = train_test_split( iris.data, iris.target, random_state=42)

gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0) 
gbrt.fit(X_train, y_train)

print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape)) 
# plot the first few entries of the decision function 
print("Decision function:\n{}".format(gbrt.decision_function(X_test)[:6, :]))

print("Argmax of decision function:\n{}".format(np.argmax(gbrt.decision_function(X_test), axis=1)))
print("Predictions:\n{}".format(gbrt.predict(X_test)))

# show the first few entries of predict_proba 
print("Predicted probabilities:\n{}".format(gbrt.predict_proba(X_test)[:6])) 
# show that sums across rows are one 
print("Sums: {}".format(gbrt.predict_proba(X_test)[:6].sum(axis=1)))

print("Argmax of predicted probabilities:\n{}".format(np.argmax(gbrt.predict_proba(X_test), axis=1))) 
print("Predictions:\n{}".format(gbrt.predict(X_test)))


logreg = LogisticRegression()

# represent each target by its class name in the iris dataset 
named_target = iris.target_names[y_train] 
logreg.fit(X_train, named_target) 
print("unique classes in training data: {}".format(logreg.classes_)) 
print("predictions: {}".format(logreg.predict(X_test)[:10])) 
argmax_dec_func = np.argmax(logreg.decision_function(X_test), axis=1) 
print("argmax of decision function: {}".format(argmax_dec_func[:10])) 
print("argmax combined with classes_: {}".format( logreg.classes_[argmax_dec_func][:10]))

