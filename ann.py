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
# Units is the number of nodes in the hidden layer (better to use k-fold cross validation)
# But you can use number nodes of input layer (11) + nodes in output layer (1) and avg them = 6
# kernel_initializer is for randomly initialize the weights according to a uniform distribution

# First hidden layer
# Input dim is the number of independent variables
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Output layer
# Units = 1 because there's only one output (if the customer exited or not)
# Use Softmax if you're dealing with a dependent variable with more than one category and units = # of categories
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# Adam is a type of stochastic gradient descent algorithms
# Stochastic gradient descent is a loss function that is based on the sum of square errors
# When you use the sigmoid activation function you have to use a logarithmic loss function
# If you have two outcomes you use binary cross entropy
# Accuracy metrics helps improve the performance of the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting classifier to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# If the y_pred is larger than 0.5, True, otherwise False
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)