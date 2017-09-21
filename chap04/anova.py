from sklearn.datasets import load_breast_cancer 
from sklearn.feature_selection import SelectPercentile 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

cancer = load_breast_cancer()

# get deterministic random numbers 
rng = np.random.RandomState(42) 
noise = rng.normal(size=(len(cancer.data), 50)) 

# add noise features to the data 
# the first 30 features are from the dataset, the next 50 are noise 
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split( X_w_noise, cancer.target, random_state=0, test_size=.5) 

# use f_classif (the default) and SelectPercentile to select 50% of features 
select = SelectPercentile(percentile=50) 
select.fit(X_train, y_train) 
# transform training set 
X_train_selected = select.transform(X_train)

print("X_train.shape: {}".format(X_train.shape)) 
print("X_train_selected.shape: {}".format(X_train_selected.shape))

mask = select.get_support() 
print(mask) 

# transform test data 
X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print("Score with all features: {:.3f}".format(lr.score(X_test, y_test)))

lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(lr.score(X_test_selected, y_test)))

# visualize the mask -- black is True, white is False 
plt.matshow(mask.reshape(1, -1), cmap='gray_r') 
plt.xlabel("Sample index")
plt.show()