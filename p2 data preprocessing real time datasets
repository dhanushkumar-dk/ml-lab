# Importing libraries
import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer

# Dataset file path
# (Ensure the 'diabetes.csv' file is in the working directory or provide the full path)
dataset_path = 'diabetes.csv'

# Data parameters
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

# Preparing the dataset using the specified file and columns
dataset = pd.read_csv(dataset_path, names=names)

# Convert dataset to a NumPy array
array = dataset.values

# Separate the array into input (X) and output (Y) components
X = array[:, 0:8]
Y = array[:, 8]

# Min-Max Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)

# Summarize transformed data
np.set_printoptions(precision=3)
print("Rescaled Data:")
print(rescaledX[0:5, :])

# Standardization
scaler = StandardScaler().fit(X)
standardizedX = scaler.transform(X)

# Summarize transformed data
np.set_printoptions(precision=3)
print("\nStandardized Data:")
print(standardizedX[0:5, :])

# Binarization
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)

# Summarize transformed data
np.set_printoptions(precision=3)
print("\nBinarized Data:")
print(binaryX[0:5, :])
