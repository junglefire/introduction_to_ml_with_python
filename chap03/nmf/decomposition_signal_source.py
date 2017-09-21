from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

S = mglearn.datasets.make_signals() 
# mix data into a 100-dimensional state 
A = np.random.RandomState(0).uniform(size=(100, 3)) 
X = np.dot(S, A.T) 
print("Shape of measurements: {}".format(X.shape))

nmf = NMF(n_components=3, random_state=42) 
S_ = nmf.fit_transform(X) 
print("Recovered signal shape: {}".format(S_.shape))

pca = PCA(n_components=3) 
H = pca.fit_transform(X)

models = [X, S, S_, H] 
names = ['Observations (first three measurements)', 'True sources', 'NMF recovered signals', 'PCA recovered signals']

fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5}, subplot_kw={'xticks': (), 'yticks': ()})

for model, name, ax in zip(models, names, axes):
    ax.set_title(name) 
    ax.plot(model[:, :3], '-')

plt.show()



