

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
import seaborn as sns
 

# Loading file into str. array 
# df = np.genfromtxt("LA_Ozone_Data.txt", dtype=int, skip_header=0, encoding=None, delimiter=",")
# ClassLabels = (df[0,])
df = pd.read_csv("LA_Ozone_Data.txt", header=None)

# Converting the panda dataframe to a numpy array: 
raw_data = df.values
data_values = raw_data[1:,:].astype(float)
class_labels = raw_data[0,:]



###################################
# 3-Data visualization
###################################

# Boxplots - Are there any Outliers?

units = ["log(g/m3)", "m", "mph", "%", "F", "m", "Pa/m", "F", "miles", "-"]

def box_plot(data_values, class_labels, units):

    rows = 3
    columns = 3
    #initiates array plotting of 3x3
    fig, axes = plt.subplots(rows, columns)
    fig.tight_layout()

    # plots the boxplot in a 3x3 array, with appropriate names and units
    count = 0
    for r in range(rows):
        for c in range(columns):
            current_boxplot = axes[r, c]
            current_boxplot.set_title(f'{class_labels[count]} [{units[count]}]')
            current_boxplot.boxplot(data_values[1:,count])

            count += 1

    plt.show()

#box_plot(data_values, class_labels,units)

# Boxplots - Are there any Outliers?

def outliers(class_labels, data_values):

    for k in range(len(data_values[0,:])):
        q1, q3= np.percentile(data_values[:,k],[25,75])
        iqr = q3 - q1
        lower_bound = q1 -(1.5 * iqr) 
        upper_bound = q3 +(1.5 * iqr)

        local_outliers = 0
        for i in range(len(data_values[:,0])):
            if data_values[i,k] <= upper_bound and data_values[i,k] >= lower_bound:
                pass
            else:
                local_outliers += 1
        
        print(f"Attribute {class_labels[k]} contains {local_outliers} outliers.")

#outliers(class_labels, data_values)


def normal_distribution(class_labels, data_values):
    
    rows = 3
    columns = 3
    bin_number = 10
    #initiates array plotting of 3x3
    fig, axes = plt.subplots(rows, columns)
    fig.tight_layout()
    
    count = 0
    for r in range(rows):
        for c in range(columns):
            #Making the 3x3 array for histogram plots
            current_histogram = axes[r, c]
            #Setting the title for each subplot
            current_histogram.set_title(f'{class_labels[count]}')
            #plotting each subplot as a histograd using the probability density function(pdf)
            current_histogram.hist(data_values[:,count], bins=bin_number, density=True, alpha=1, color='r')
            #determining the mean and std for each attributes, and using this to plot a normal distrubution
            mean, std = norm.fit(data_values[:,count])
            normal_data = np.linspace(min(data_values[:,count]), max(data_values[:,count]), 80)
            current_histogram.plot(normal_data, norm.pdf(normal_data, mean, std), 'b', linewidth=2)

            #adding a value to count, to go to next subbplot
            count += 1

    plt.show()

#normal_distribution(class_labels, data_values)




def correlation_heatmap(data_values, class_labels):
    # checking correlation using heatmap
    data_frame = pd.DataFrame(data_values[:,:9], columns = class_labels[:9])
    correlation_mat = data_frame.corr()
    mask = np.triu(np.ones_like(correlation_mat, dtype=bool))
    sns.heatmap(correlation_mat, mask = mask, annot = True)
    plt.show()

correlation_heatmap(data_values, class_labels)