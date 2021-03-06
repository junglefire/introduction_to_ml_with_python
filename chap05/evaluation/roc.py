from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_curve

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

# Use more data points for a smoother curve 
# X, y = make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2], random_state=22) 
X, y = make_blobs(n_samples=4000, centers=2, cluster_std=[7.0, 2], random_state=22) 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 

svc = SVC(gamma=.05).fit(X_train, y_train) 
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test)) 

# find threshold closest to zero 
close_zero = np.argmin(np.abs(thresholds)) 

# RandomForestClassifier has predict_proba, but not decision_function
rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)

precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])

plt.plot(precision, recall, label="svc")
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10, label="threshold zero svc", fillstyle="none", c='k', mew=2)
plt.plot(precision_rf, recall_rf, label="rf")

close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5)) 
plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^', c='k', markersize=10, label="threshold 0.5 rf", fillstyle="none", mew=2)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")
plt.show()

print("f1_score of random forest: {:.3f}".format(f1_score(y_test, rf.predict(X_test)))) 
print("f1_score of svc: {:.3f}".format(f1_score(y_test, svc.predict(X_test))))

ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1]) 
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
print("Average precision of random forest: {:.3f}".format(ap_rf))
print("Average precision of svc: {:.3f}".format(ap_svc))



