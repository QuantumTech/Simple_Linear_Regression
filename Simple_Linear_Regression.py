# Simple linear Regression .
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Importing the libraries

# 'np' is the numpy shortcut!
# 'plt' is the matplotlib shortcut!
# 'pd' is the pandas shortcut!

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Importing the dataset

# Data set is creating the data frame of the 'Salary_Data.csv' file
# Features (independent variables) = The columns the predict the dependent variable
# Dependent variable = The last column
# 'X' = The matrix of features (country, age, salary)
# 'Y' = Dependent variable vector (purchased (last column))
# '.iloc' = locate indexes[rows, columns]
# ':' = all rows (all range)
# ':-1' = Take all the columns except the last one
# '.values' = taking all the values

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
#print(X)
y = dataset.iloc[:, -1].values # NOTICE! .iloc[all the rows, only the last column]
#print(y)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Splitting the dataset into the Training set and Test set

# Note to self! = Split the data before feature scaling!
# Test set = future data
# Feature scaling = scaling the features so that they all take values in the same scale
# 80/20 split
# 'test_size' = 20% for the test set

# 'X_train' The matrix of the features of the training set
# 'X_test' The matrix of the features of the test set
# 'y_train' The dependent variable of the training set
# 'y_test' The dependent variable of the test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#print(X_train) # The matrix of the features of the training set
#print(X_test) # The matrix of the features of the test set
#print(y_train) # The dependent variable of the training set
#print(y_test) # The dependent variable of the test set

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Simple Linear Regression.

# Training the Simple Linear Regression model on the Training set

# 'X_train' The matrix of the features of the training set
# 'y_train' The dependent variable of the training set

from sklearn.linear_model import LinearRegression #Importing the linear regression model from the 'sklearn' library.
regressor = LinearRegression() #Object of the 'linearRegression' class above.
regressor.fit(X_train, y_train) #The fit method is being used to train the regression model.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Simple Linear Regression.

# Predicting the Test set results
# Reminder! = 'regressor' is an object
# 'X_test' The matrix of the features of the test set

y_pred = regressor.predict(X_test) # Returns a vector of predictions.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Simple Linear Regression.

# Visualising the TRAINING set results

# Reminder! 'plt' is short hand for 'matplotlib'
# 'scatter' = Marks the scatter plots on the 2D data.

plt.scatter(X_train, y_train, color = 'red') #The coordinates for the scatter plot.
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # Colour of the regression line
plt.title('Salary vs Experience (Training set)') # Title
plt.xlabel('Years of Experience!') # Horizontal axis
plt.ylabel('Salary') # Vertical axis
plt.show() # Show function to display the graphics
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Simple Linear Regression.

# Visualising the TEST set results

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()