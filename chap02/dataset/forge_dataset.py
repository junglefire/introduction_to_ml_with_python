import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

# generate dataset
# X是sample的二维数组，每个维度代表一个feature
# y是一维数组，代表每个sample的label
X, y = mglearn.datasets.make_forge()

# plot dataset
# 分别取X的第一个维度和第二个维度作为横坐标和纵坐标
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

plt.legend(["Class 0", "Class 1"], loc=4) 
plt.xlabel("First feature") 
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
plt.show()