import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier

from .datasets import make_forge
from .plot_helpers import discrete_scatter

# 使用knn算法训练forge数据集
def plot_knn_classification(n_neighbors=1):
    X, y = make_forge()

    X_test = np.array([[8.2, 3.66214339], [9.9, 3.2], [11.2, .5]])
    # euclidean_distances()计算每个训练数据到测试数据的欧几里德距离 
    dist = euclidean_distances(X, X_test)
    # np.argsort函数返回的是数组值从小到大的索引值
    closest = np.argsort(dist, axis=0)

    for x, neighbors in zip(X_test, closest.T):
        # print("x=", x)
        # print("neighbors=", neighbors)
        for neighbor in neighbors[:n_neighbors]:
            # print("neighbor=", neighbor)
            plt.arrow(x[0], x[1], X[neighbor, 0]-x[0], X[neighbor, 1]-x[1], head_width=0, fc='k', ec='k')

    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    test_points = discrete_scatter(X_test[:, 0], X_test[:, 1], clf.predict(X_test), markers="*")
    training_points = discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(training_points + test_points, ["training class 0", "training class 1", "test pred 0", "test pred 1"])
