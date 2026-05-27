import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../data/loan_data.csv")

print(df.head())

sns.countplot(x="loan_status", data=df)

plt.show()