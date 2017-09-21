from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import graphviz

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

tree = mglearn.plots.plot_tree_not_monotone()
display(tree)

