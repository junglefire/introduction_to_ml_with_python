from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

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
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10, label="threshold zero", fillstyle="none", c='k', mew=2)
plt.plot(precision, recall, label="precision recall curve") 
plt.xlabel("Precision") 
plt.ylabel("Recall")
plt.show()
