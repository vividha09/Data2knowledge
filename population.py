import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/kaggle/input/india-census/india-districts-census-2011.csv")
dfmax=df['Population'].max()
print(dfmax)
df.loc[df['Population']==dfmax]
