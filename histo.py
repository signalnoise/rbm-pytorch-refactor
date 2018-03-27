import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df=pd.read_csv("weights.txt", dtype=np.float32)
df.plot.hist(bins=100)
print(df.as_matrix().mean())
plt.show()
