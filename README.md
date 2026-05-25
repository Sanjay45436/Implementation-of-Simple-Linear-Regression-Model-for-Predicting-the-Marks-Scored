# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Load the dataset and separate input and output variables.
2.Split the data into training and testing sets.
3.Train the linear regression model using the training data.
4.Predict the output for the test data and evaluate the results..
```
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
   Developed by: LINCE GIYO L
RegisterNumber: 212225100023


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read dataset
df = pd.read_csv('student_scores.csv')

# Display first and last rows
print(df.head())
print(df.tail())

# Independent variable (Hours)
X = df.iloc[:, :-1].values

# Dependent variable (Scores)
Y = df.iloc[:, 1].values

print(X)
print(Y)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0
)

# Train model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Prediction
Y_pred = regressor.predict(X_test)

print("Predicted Values:")
print(Y_pred)

print("Actual Values:")
print(Y_test)

# Training set graph
plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")

plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

plt.show()

# Test set graph
plt.scatter(X_test, Y_test, color="purple")
plt.plot(X_train, regressor.predict(X_train), color="blue")

plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

plt.show()

# Error calculations
mse = mean_squared_error(Y_test, Y_pred)
print("MSE =", mse)

mae = mean_absolute_error(Y_test, Y_pred)
print("MAE =", mae)

rmse = np.sqrt(mse)
print("RMSE =", rmse)
```
## Output:
TRAINING SET:
<img width="871" height="573" alt="image" src="https://github.com/user-attachments/assets/cb238b3e-ac3e-4c31-a7db-12123b2ab753" />

TEST SET:
<img width="840" height="549" alt="image" src="https://github.com/user-attachments/assets/80d27968-6b1e-4548-8c1c-7cecbc1b3e4a" />
  
<img width="260" height="83" alt="image" src="https://github.com/user-attachments/assets/efc1fbb6-8738-422c-b5a3-a7d07a3b4b68" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
