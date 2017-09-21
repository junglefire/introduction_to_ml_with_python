from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

digits = load_digits()

# build a PCA model 
pca = PCA(n_components=2) 
pca.fit(digits.data) 

# transform the digits data onto the first two principal components 
digits_pca = pca.transform(digits.data) 
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]

plt.figure(figsize=(10, 10)) 
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max()) 
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max()) 

for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), color = colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9}) 

plt.xlabel("First principal component") 
plt.ylabel("Second principal component")
plt.show()



