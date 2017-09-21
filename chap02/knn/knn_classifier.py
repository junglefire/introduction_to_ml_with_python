from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 训练模型
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

# 对测试数据集进行判断
print("Test set predictions: {}".format(clf.predict(X_test)))
# 模型的准确率
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))