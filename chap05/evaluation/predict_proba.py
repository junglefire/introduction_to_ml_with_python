from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

X, y = mglearn.datasets.make_blobs(n_samples=(400, 50), centers=2, cluster_std=[7.0, 2], random_state=22) 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)

# mglearn.plots.plot_decision_threshold()
# plt.show()

print("threshold = 0.0")
print(classification_report(y_test, svc.predict(X_test)))

print("threshold = -0.8")
y_pred_lower_threshold = svc.decision_function(X_test) > -.8
print(classification_report(y_test, y_pred_lower_threshold))

