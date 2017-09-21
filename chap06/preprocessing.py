from sklearn.svm import SVC 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

# load and split the data 
cancer = load_breast_cancer() 
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# compute minimum and maximum on the training data 
scaler = MinMaxScaler().fit(X_train)

# rescale the training data 
X_train_scaled = scaler.transform(X_train) 
svm = SVC() 

# learn an SVM on the scaled training data 
svm.fit(X_train_scaled, y_train) 

# scale the test data and score the scaled data 
X_test_scaled = scaler.transform(X_test) 
print("Test score: {:.2f}".format(svm.score(X_test_scaled, y_test)))

# for illustration purposes only, don't use this code!
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]} 

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Best set score: {:.2f}".format(grid.score(X_test_scaled, y_test)))
print("Best parameters: ", grid.best_params_)