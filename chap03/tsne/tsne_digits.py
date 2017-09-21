from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

digits = load_digits()

tsne = TSNE(random_state=42) 
# use fit_transform instead of fit, as TSNE has no transform method 
digits_tsne = tsne.fit_transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]

plt.figure(figsize=(10, 10)) 
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1) 
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1) 

for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), color = colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9}) 

plt.xlabel("t-SNE feature 0")
plt.xlabel("t-SNE feature 1")
plt.show()



