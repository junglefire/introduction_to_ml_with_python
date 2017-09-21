from sklearn.feature_selection import SelectFromModel 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
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

select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="median")
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_l1.shape: {}".format(X_train_l1.shape))

X_test_l1 = select.transform(X_test) 
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test) 
print("Test score: {:.3f}".format(score))

mask = select.get_support() 
# visualize the mask -- black is True, white is False 
plt.matshow(mask.reshape(1, -1), cmap='gray_r') 
plt.xlabel("Sample index")
plt.show()