from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

cancer = load_breast_cancer()
scaler = StandardScaler() 
scaler.fit(cancer.data) 
X_scaled = scaler.transform(cancer.data)

# keep the first two principal components of the data 
pca = PCA(n_components=2) 

# fit PCA model to breast cancer data 
pca.fit(X_scaled) 

# transform data onto the first two principal components 
X_pca = pca.transform(X_scaled) 

print("Original shape: {}".format(str(X_scaled.shape))) 
print("Reduced shape: {}".format(str(X_pca.shape)))

print("PCA component shape: {}".format(pca.components_.shape))
print("PCA components:\n{}".format(pca.components_))

# plot first vs. second principal component, colored by class 
plt.matshow(pca.components_, cmap='viridis') 
plt.yticks([0, 1], ["First component", "Second component"]) 
plt.colorbar() 
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left') 
plt.xlabel("Feature") 
plt.ylabel("Principal components")
plt.show()

