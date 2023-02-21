

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Loading file into str. array 
# df = np.genfromtxt("LA_Ozone_Data.txt", dtype=int, skip_header=0, encoding=None, delimiter=",")
# ClassLabels = (df[0,])
df = pd.read_csv("LA_Ozone_Data.txt", header=None)

# Converting the panda dataframe to a numpy array: 
raw_data = df.values
data_values = raw_data[1:,:].astype(np.float)
class_labels = raw_data[0,:]




###################################
# 3-Data visualization
###################################

# Exercise 3.1.1 - Boxplots

units = ["log(g/m3)", "m", "mph", "%", "F", "m", "Pa/m", "F", "miles", "-"]

def box_plot(data_values, class_labels, units):

    rows = 3
    columns = 3
    fig, axes = plt.subplots(rows, columns)
    fig.tight_layout()

    count = 0
    for r in range(rows):
        for c in range(columns):
            current_boxplot = axes[r, c]
            current_boxplot.set_title(f'{class_labels[count]} [{units[count]}]')
            current_boxplot.boxplot(data_values[1:,count])

            count += 1

    plt.show()

box_plot(data_values, class_labels,units)
