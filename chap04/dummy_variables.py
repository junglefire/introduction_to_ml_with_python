import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

# The file has no headers naming the columns, so we pass header=None 
# and provide the column names explicitly in "names" 
data = pd.read_csv("./data/adult.data", header=None, index_col=False, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']) 

# For illustration purposes, we only select some of the columns 
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']] 

# IPython.display allows nice output formatting within the Jupyter notebook 
# display(data.head())
print(data.head())
print("\n\n")

print(data.gender.value_counts())
print("\n\n")

print("Original features:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("Features after get_dummies:\n", list(data_dummies.columns))
print("\n\n")

# print(data_dummies.head())

features = data_dummies.ix[:, 'age':'occupation_ Transport-moving']

# Extract NumPy arrays 
X = features.values
y = data_dummies['income_ >50K'].values 
print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
logreg = LogisticRegression() 
logreg.fit(X_train, y_train)
print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))
