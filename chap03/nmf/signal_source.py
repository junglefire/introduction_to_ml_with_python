import matplotlib.pyplot as plt

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

S = mglearn.datasets.make_signals() 
plt.figure(figsize=(6, 1)) 
plt.plot(S, '-') 
plt.xlabel("Time") 
plt.ylabel("Signal")
plt.show()



