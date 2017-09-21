import matplotlib.pyplot as plt
import pandas as pd

# 导入mglearn模块
import sys
sys.path.append("../")
import mglearn

ram_prices = pd.read_csv("data/ram_price.csv")
plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte")
plt.show()
