from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import pandas as pd
import numpy as np
import mglearn

iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}\n".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...\n")
print("Target names: {}\n".format(iris_dataset['target_names']))
print("Feature names: \n{}\n".format(iris_dataset['feature_names']))
print("Type of data: {}\n".format(type(iris_dataset['data'])))
print("Shape of data: {}\n".format(iris_dataset['data'].shape))
print("First five columns of data:\n{}\n".format(iris_dataset['data'][:5]))
print("Type of target: {}\n".format(type(iris_dataset['target'])))
print("Shape of target: {}\n".format(iris_dataset['target'].shape))
print("Target:\n{}\n".format(iris_dataset['target']))

print(">>>>>>>>>>>>>>>>> SPLIT DATASET <<<<<<<<<<<<<<<<<<<<<")
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print("X_train shape: {}\n".format(X_train.shape)) 
print("y_train shape: {}\n".format(y_train.shape))
print("X_test shape: {}\n".format(X_test.shape)) 
print("y_test shape: {}\n".format(y_test.shape))

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names 
# iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# create a scatter matrix from the dataframe, color by y_train
# grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

print(">>>>>>>>>>>>>>>>> KNN TEST CASE <<<<<<<<<<<<<<<<<<<<<")
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("kNN object is {}\n".format(knn))

X_new = np.array([[5, 2.9, 1, 0.2]]) 
print("X_new.shape: {}\n".format(X_new.shape))

prediction = knn.predict(X_new) 
print("Prediction: {}\n".format(prediction)) 
print("Predicted target name: {}\n".format(iris_dataset['target_names'][prediction]))

print(">>>>>>>>>>>>>>>>> EVALUATING THE MODEL <<<<<<<<<<<<<<<<<<<<<")
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}\n".format(y_pred))
print("Test set score: {:.2f}\n".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}\n".format(knn.score(X_test, y_test)))

