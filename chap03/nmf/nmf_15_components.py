from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import numpy as np

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7) 
print("people.images.shape: {}".format(people.images.shape)) 
print("Number of classes: {}".format(len(people.target_names)))

# count how often each target appears 
counts = np.bincount(people.target)
print("counts=", counts)

# to make the data less skewed, we will only take up to 50 
# images of each person
mask = np.zeros(people.target.shape, dtype=np.bool)

for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# scale the grayscale values to be between 0 and 1 
# instead of 0 and 255 for better numeric stability
X_people = X_people / 255.

# split the data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0) 

image_shape = people.images[0].shape
nmf = NMF(n_components=15, random_state=0) 
nmf.fit(X_train) 
X_train_nmf = nmf.transform(X_train) 
X_test_nmf = nmf.transform(X_test)

fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})

for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title("{}. component".format(i))

plt.show()