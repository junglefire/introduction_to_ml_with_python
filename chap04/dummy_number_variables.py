import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

# create a DataFrame with an integer feature and a categorical string feature 
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1], 'Categorical Feature': ['socks', 'fox', 'socks', 'box']}) 
print(demo_df)
print("\n\n")

d1 = pd.get_dummies(demo_df)
print(d1)
print("\n\n")

demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str) 
d2 = pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature'])
print(d2)
print("\n\n")


