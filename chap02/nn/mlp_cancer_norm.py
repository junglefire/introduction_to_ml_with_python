from sklearn.neural_network import MLPClassifier 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

# compute the mean value per feature on the training set
mean_on_train = X_train.mean(axis=0)
# compute the standard deviation of each feature on the training set 
std_on_train = X_train.std(axis=0)
# subtract the mean, and scale by inverse standard deviation
# afterward, mean=0 and std=1
X_train_scaled = (X_train - mean_on_train) / std_on_train

# use THE SAME transformation (using training mean and std) on the test set
X_test_scaled = (X_test - mean_on_train) / std_on_train
mlp = MLPClassifier(random_state=0)
mlp.fit(X_train_scaled, y_train)

print("(default) Accuracy on training set: {:.3f}".format( mlp.score(X_train_scaled, y_train)))
print("(default) Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("(max_iter=1000) Accuracy on training set: {:.3f}".format( mlp.score(X_train_scaled, y_train)))
print("(max_iter=1000) Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)
print("(max_iter=1000, alpha=1) Accuracy on training set: {:.3f}".format( mlp.score(X_train_scaled, y_train)))
print("(max_iter=1000, alpha=1) Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))


