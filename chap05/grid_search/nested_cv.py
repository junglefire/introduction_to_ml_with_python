from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid, StratifiedKFold
import numpy as np

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

iris = load_iris()

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5), iris.data, iris.target, cv=5)
print("Cross-validation scores: ", scores)
print("Mean cross-validation score: ", scores.mean())

def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_scores = [] 
    # for each split of the data in the outer cross-validation 
    # (split method returns indices) 
    for training_samples, test_samples in outer_cv.split(X, y):
        # find best parameter using inner cross-validation 
        best_parms = {}
        best_score = -np.inf
        # iterate over parameters 
        for parameters in parameter_grid:
            # accumulate score over inner splits 
            cv_scores = [] 
            # iterate over inner cross-validation 
            for inner_train, inner_test in inner_cv.split( X[training_samples], y[training_samples]):
                # build classifier given parameters and training data 
                clf = Classifier(**parameters) 
                clf.fit(X[inner_train], y[inner_train]) 
                # evaluate on inner test set 
                score = clf.score(X[inner_test], y[inner_test]) 
                cv_scores.append(score) 
            # compute mean score over inner folds 
            mean_score = np.mean(cv_scores) 
            if mean_score > best_score:
                # if better than so far, remember parameters 
                best_score = mean_score 
                best_params = parameters 
        # build classifier on best parameters using outer training set 
        clf = Classifier(**best_params)
        clf.fit(X[training_samples], y[training_samples])
        # evaluate
        outer_scores.append(clf.score(X[test_samples], y[test_samples]))
    return np.array(outer_scores)

scores = nested_cv(iris.data, iris.target, StratifiedKFold(5), StratifiedKFold(5), SVC, ParameterGrid(param_grid)) 
print("Cross-validation scores: {}".format(scores))

