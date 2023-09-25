# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values

3.Import linear regression from sklearn

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Archana k
RegisterNumber: 212222240011

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)  
*/
```

## Output:

## df.head()

![265592191-b153bb27-164d-43de-b78f-2d687047915e](https://github.com/22009150/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118708624/ae3c11e0-8714-4dff-81ab-dac8b9187ebb)

## df.tail()

![265592658-cc40deec-122c-41f7-a1dd-99c7120ac978](https://github.com/22009150/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118708624/5e435fb5-de13-41c7-868d-f87e494a4823)

## Array value of x

![265592749-203ddaa2-3258-4d60-b305-20aad1232536](https://github.com/22009150/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118708624/ea98c723-92ae-44a6-a062-db3b1461dd88)

## Array value of y

![265592923-aa4401bb-281b-44bf-892b-493f082afb05](https://github.com/22009150/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118708624/e6dc2753-7490-406c-a745-dc30bc3d6121)

## Array value of y test

![265593091-c4366894-7578-41ad-ab91-1bc3f3fa27d4](https://github.com/22009150/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118708624/d350771b-75d0-4544-a400-49dc6ebd64c9)

## Training set Graph

![265593376-543d3531-59cc-401b-8e6e-85d2b938c49a](https://github.com/22009150/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118708624/5bd42191-f07e-4673-849a-0f72389f7782)

## Test set graph

![265593409-b9242df8-4530-41e4-9819-76840f3ca757](https://github.com/22009150/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118708624/a3c24f8f-101e-46fa-8ffb-716ecb2b4e6f)

## Value of MSE,MAE, and RMSE

![265593520-6f20014e-2e8b-4218-9ffd-f6f174d43a7d](https://github.com/22009150/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118708624/cea41ed7-4a04-4e04-a5bb-4795c0b7459c)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
