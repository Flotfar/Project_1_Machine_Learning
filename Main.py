

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
import seaborn as sns
from scipy.linalg import svd

# Loading file into str. array 
# df = np.genfromtxt("LA_Ozone_Data.txt", dtype=int, skip_header=0, encoding=None, delimiter=",")
# ClassLabels = (df[0,])
df = pd.read_csv(r"C:\Users\jkdah\git\Project-1---Machine-Learning\LA_Ozone_Data.txt", header=None)

# Converting the panda dataframe to a numpy array: 
raw_data = df.values
data_values = raw_data[1:,:].astype(float)
class_labels = raw_data[0,:]



###############################################
# 3-Data visualization
###############################################

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

<<<<<<< HEAD
#correlation_heatmap(data_values, class_labels)


def pca_cummulative(data_values):

    N = len(data_values[:,0])

    columns_to_delete = [2, 5, 6, 8]
    plot_titles = ['Variance explained by principal components (all attributes)',
                   'Variance explained by principal components (selected attributes)']

    pc_coefficients = [5,2]

    #Removing doy from data_values
    X_all = np.delete(data_values, -1, 1)
    #Removing un-selected data from X_all
    X_selected = np.delete(X_all, columns_to_delete, 1)

    #For loop over the to datasets
    for i, X in enumerate([X_all, X_selected]):
        
        #Normalizing the data
        Y = (X - X.mean(axis=0)*np.ones((N,1))) / X.std(axis=0)*np.ones((N,1))

        #Using exercise 2_1_3 to plot varience explained
        # PCA by computing SVD of Y 
        U,S,V = svd(Y,full_matrices=False)

        # Compute variance explained by principal components
        rho = (S*S) / (S*S).sum() 

        #printing the pc_coefficients for all (5) and selected attributes (2)
        print(V[:pc_coefficients[i]])

        #Defining the threshold
        threshold = 0.9

        # Plot variance explained in two subplots, for selected and all data
        plt.subplot(1, 2, i + 1)
        plt.plot(range(1,len(rho)+1),rho,'x-')
        plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
        plt.plot([1,len(rho)],[threshold, threshold],'k--')
        plt.title(plot_titles[i]);
        plt.xlabel('Principal component');
        plt.ylabel('Variance explained');
        plt.legend(['Individual','Cumulative','Threshold'])
        plt.grid()


    #plotting
    plt.show()

pca_cummulative(data_values)


"""
def pca_bar_chart(data_values, class_labels):

    N = len(data_values[:,0])

    columns_to_delete = [2, 5, 6, 8]
    plot_titles = ['PC coefficients (all attributes)',
                   'PC coefficients (selected attributes)']

    #Removing doy from data_values
    X_all = np.delete(data_values, -1, 1)
    label_all = np.delete(class_labels, -1, 0)
    #Removing un-selected data from X_all
    X_selected = np.delete(X_all, columns_to_delete, 1)
    label_selected = np.delete(class_labels, columns_to_delete, 0)
    pc_coefficients = [5,2]

    

    #For loop over the to datasets
    for i, X in enumerate([X_all, X_selected]):
        
        #Normalizing the data
        Y = (X - X.mean(axis=0)*np.ones((N,1))) / X.std(axis=0)*np.ones((N,1))

        #Using exercise 2_1_3 to plot varience explained
        # PCA by computing SVD of Y 
        U,S,V = svd(Y,full_matrices=False)
        V_current = V[:pc_coefficients[i]]
        # Plot variance explained in two subplots, for selected and all data
        plt.subplot(1, 2, i + 1)
        
        # create data
        
        # plot data in grouped manner of bar type
        if i == 0:

            x = np.arange(pc_coefficients[i])
            width = 0.2

            plt.bar(x-0.4, V_current[0], width, color='red')
            plt.bar(x-0.2, V_current[1], width, color='cyan')
            plt.bar(x, V_current[2], width, color='orange')
            plt.bar(x+0.2, V_current[3], width, color='green')
            plt.bar(x+0.4, V_current[4], width, color='magenta')
            plt.xticks(x, label_all)
            plt.xlabel("Attributes")
            plt.ylabel("PC coefficients")
            plt.legend(["PC1", "PC2","PC3","PC4","PC5"])

    #plotting
    plt.show()

pca_bar_chart(data_values, class_labels)
"""
=======
correlation_heatmap(data_values, class_labels)



###############################################
# PCA1 & PCA2 plotting 
###############################################

# creating month 'moy' column into the array for categorization
doy = raw_data[1:, 9]
days = [eval(i) for i in doy]
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
month = ["moy"]

for day in days:
    for i, days_in_this_month in enumerate(days_in_month):
        if day <= days_in_this_month:
            month.append(i+1)
            break
        else:
            day -= days_in_this_month

with_months = np.column_stack((raw_data, month))

# extracting non correlating attributes and related column:
columns = [2, 5, 6, 8, 9]
data = np.delete(with_months, np.s_[2, 5, 6, 8, 9], 1)




>>>>>>> df79cee47f7a6577ec4f5282ffddb84082409408
