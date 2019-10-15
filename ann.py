# Artificial Neural Networks

# Step 1 - data prep
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Columns 3 through 12
X = dataset.iloc[:, 3:13].values

# Dependent variable is column 12
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Converts France, Spain to 0, 1, etc.
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Converts Male/Female to 0, 1
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

# Creates the dummy variables
transformer = ColumnTransformer(
    transformers=[
        ("OneHot", OneHotEncoder(), [1])
    ],
    remainder='passthrough'
)

X = transformer.fit_transform(X.tolist())

# All the rows and all the columns except the first
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 2 - create ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# Create the sequence of layers to init ANN
classifier = Sequential()

# Add the input layer
# Output dim is the number of nodes in the hidden layer (better to use k-fold cross validation)
# Can use number nodes of input layer (11) + nodes in output layer (1) and avg them = 6

# Init is for randomly initialize the weights according to a uniform distribution
# Input dim is the number of independent variables
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# # Fitting classifier to the Training set
# # Create your classifier here

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)