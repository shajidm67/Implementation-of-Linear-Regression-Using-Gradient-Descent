# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
/*
Program to implement the linear regression using gradient descent.
Developed by: A.MOHAMED SHAJID
RegisterNumber:  212225040243
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("50_Startups.csv")

X = data[["R&D Spend", "Administration", "Marketing Spend"]].values #convert a DataFrame or Series into a NumPy array.
y = data["Profit"].values

# Feature Scaling (VERY IMPORTANT)

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Parameters

m, n = X.shape          # m = samples, n = features
w = np.zeros(n)         # weights (w1, w2, w3)
b = 0.0                 # bias
alpha = 0.01            # learning rate
epochs = 1000

losses = []

# Gradient Descent
for i in range(epochs):
    # Prediction
    y_hat = np.dot(X, w) + b

    # Loss (Mean Squared Error)
    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)

    # Gradients
    dw = (2/m) * np.dot(X.T, (y_hat - y))
    db = (2/m) * np.sum(y_hat - y)

    # Update parameters
    w = w - alpha * dw
    b = b - alpha * db

# -----------------------
# Results
# -----------------------
print("Final Weights:", w)
print("Final Bias:", b)

# -----------------------
# Loss vs Iterations
# -----------------------
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations (Multiple Linear Regression)")
plt.show()

## Output:
<img width="882" height="622" alt="image" src="https://github.com/user-attachments/assets/6240dc38-61c8-4d5f-b747-06125c6ccfcb" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
