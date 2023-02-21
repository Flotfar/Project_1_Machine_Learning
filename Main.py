

import numpy as np
import pandas as pd


# Loading file into str. array 
# df = np.genfromtxt("LA_Ozone_Data.txt", dtype=int, skip_header=0, encoding=None, delimiter=",")
# ClassLabels = (df[0,])
df = pd.read_csv("LA_Ozone_Data.txt", header=None)

# Converting the panda dataframe to a numpy array: 
raw_data = df.values 
classLabels = raw_data[0,:]

