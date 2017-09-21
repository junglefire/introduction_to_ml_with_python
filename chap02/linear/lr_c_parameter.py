from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42) 
logreg = LogisticRegression().fit(X_train, y_train)
print("(LR, c=1.0) Training set score: {:.3f}".format(logreg.score(X_train, y_train))) 
print("(LR, c=1.0) Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("(LR, c=100) Training set score: {:.3f}".format(logreg100.score(X_train, y_train))) 
print("(LR, c=100) Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("(LR, c=0.01) Training set score: {:.3f}".format(logreg001.score(X_train, y_train))) 
print("(LR, c=0.01) Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.show()
