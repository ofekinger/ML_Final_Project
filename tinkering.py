import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.show()

# Import dataset
df = pd.read_csv("feature_data.csv")

df.head()


plot_corr(df, len(df.columns))

