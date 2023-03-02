

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
import seaborn as sns
from scipy.linalg import svd

# Loading file into str. array
df = df = pd.read_csv("LA_Ozone_Data.txt", header=None)

# Converting the panda dataframe to a numpy array: 
raw_data = df.values
data_values = raw_data[1:,:].astype(float)
class_labels = raw_data[0,:]

units = ["log(g/m3)", "m", "mph", "%", "F", "m", "Pa/m", "F", "miles", "-"]

# Categorizing the datapoints in raw_data with quantile placement
doy = raw_data[1:, 9]
days = [eval(i) for i in doy]
days_in_month = [31+28+31, 30+31+30, 31+31+30, 31+30+31]
month = []

for day in days:
    for i, days_in_this_month in enumerate(days_in_month):
        if day <= days_in_this_month:
            month.append(i+1)
            break
        else:
            day -= days_in_this_month

##########################################################
# Attribute Boxplot
##########################################################


def box_plot(values, labels, units):

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
            current_boxplot.set_title(f'{labels[count]} [{units[count]}]')
            current_boxplot.boxplot(values[1:,count])

            count += 1

    plt.show()


#### Detecting outliers from the attributes ########
def outliers(values, labels):

    for k in range(len(values[0,:])):
        q1, q3= np.percentile(values[:,k],[25,75])
        iqr = q3 - q1
        lower_bound = q1 -(1.5 * iqr) 
        upper_bound = q3 +(1.5 * iqr)

        local_outliers = 0
        for i in range(len(values[:,0])):
            if values[i,k] <= upper_bound and values[i,k] >= lower_bound:
                pass
            else:
                local_outliers += 1
        
        print(f"Attribute {labels[k]} contains {local_outliers} outliers.")


##########################################################
# Attribute Normal destribution
##########################################################

def normal_distribution(values, labels):
    
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
            current_histogram.set_title(f'{labels[count]}')
            #plotting each subplot as a histograd using the probability density function(pdf)
            current_histogram.hist(values[:,count], bins=bin_number, density=True, alpha=1, color='r')
            #determining the mean and std for each attributes, and using this to plot a normal distrubution
            mean, std = norm.fit(values[:,count])
            normal_data = np.linspace(min(values[:,count]), max(values[:,count]), 80)
            current_histogram.plot(normal_data, norm.pdf(normal_data, mean, std), 'b', linewidth=2)

            #adding a value to count, to go to next subbplot
            count += 1

    plt.show()


##########################################################
# Attribute corelation heatmap
##########################################################

def correlation_heatmap(values, labels):
    # checking correlation using heatmap
    data_frame = pd.DataFrame(values[:,:9], columns = labels[:9])
    correlation_mat = data_frame.corr()
    mask = np.triu(np.ones_like(correlation_mat, dtype=bool))
    sns.heatmap(correlation_mat, mask = mask, annot = True)
    plt.show()



##########################################################
# Principal Component Analysis (PCA)
##########################################################

#Subtrackting doy from the dataset to reduce noise:
X = np.delete(data_values, np.s_[9], 1)

# Normalizing the data:
N = len(data_values[:,0])
Y = (X - X.mean(axis=0)*np.ones((N,1))) / X.std(axis=0)*np.ones((N,1))

# Running the Single Value Decompositioning (SVD)
U, S, V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()



##### Visualizing the coherance of the PC's ########## 
def pca_cummulative(input):

    #Defining data-set quantiles
    Q1 = 0.25
    Q2 = 0.50
    Q3 = 0.75

    # Plot variance explained in two subplots, for selected and all data
    plt.figure()
    plt.plot(range(1,len(input)+1),input,'x-')
    plt.plot(range(1,len(input)+1),np.cumsum(input),'o-')
    plt.plot([0,len(input)],[Q1, Q1],':', color='r', linewidth=1)
    plt.plot([0,len(input)],[Q2, Q2],':', color='r', linewidth=1)
    plt.plot([0,len(input)],[Q3, Q3],':', color='r', linewidth=1)
    plt.title('Variance explained by principal components (all attributes)');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Quantiles'])
    plt.grid()

    #plotting
    plt.show()



##### plotting the PCA aginst the 3 most influencial PC's ##########
def pca_visualization(input):

    fig1 = plt.figure(figsize=(10,8), facecolor='w')
    ax = fig1.add_subplot(projection='3d')

    for i in range(input.shape[0]):
        x = V[0,:] @ input[i,1:].T
        y = V[1,:] @ input[i,1:].T
        z = V[2,:] @ input[i,1:].T

        if month[i] == 1:
            ax.scatter(x,y,z, marker='o', color='gold', s=20)
        elif month[i] == 2:
            ax.scatter(x,y,z, marker='o', color='dodgerblue', s=20)
        elif month[i] == 3:
            ax.scatter(x,y,z, marker='o', color='limegreen', s=20)
        else:
            ax.scatter(x,y,z, marker='o', color='darkorange', s=20)

    ax.set_xlabel('\nPC1', fontsize = 15, linespacing=1)
    ax.set_ylabel('\nPC2', fontsize = 15, linespacing=2)
    ax.set_zlabel('\nPC3 ', fontsize = 15, linespacing=2)
    ax.view_init(25,-50)
    # creating dummy plot for legend applyance: 
    proxy1 = plt.Line2D([0],[0], linestyle="none", color='gold', marker = 'o')
    proxy2 = plt.Line2D([0],[0], linestyle="none", color='dodgerblue', marker = 'o')
    proxy3 = plt.Line2D([0],[0], linestyle="none", color='limegreen', marker = 'o')
    proxy4 = plt.Line2D([0],[0], linestyle="none", color='darkorange', marker = 'o')
    ax.legend([proxy1, proxy2, proxy3, proxy4], ['Q1', 'Q2', 'Q3', 'Q4' ], numpoints = 1)

    plt.show()


##### printing the plots ##########
box_plot(data_values, class_labels, units)
outliers(data_values, class_labels)
normal_distribution(data_values, class_labels)
correlation_heatmap(data_values, class_labels)
pca_cummulative(rho)
pca_visualization(data_values)


