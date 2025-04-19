# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Organize the dataset into a structured format, such as a CSV file or a DataFrame, for easy handling and analysis.

2. Apply a Simple Linear Regression model to fit the training data and identify the relationship between the independent and dependent variables.

3. Use the trained model to make predictions on the test dataset and assess its generalization capability.

4. Evaluate the model's performance using standard regression metrics, including Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), to measure prediction accuracy.



## Program:
``` 
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PREETHI D 
RegisterNumber: 212224040250
```
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df = pd.read_csv('student_scores.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/14e2bd5f-1c75-453f-a9f7-2308fe9dfc37)

``` python

df.tail()

```
![image](https://github.com/user-attachments/assets/fb19fce5-b1fc-4a4b-a3a9-7d4a62523602)

``` python

x = df.iloc[:,:-1].values
x

```
![image](https://github.com/user-attachments/assets/439b872f-9c23-442a-94cb-5149ddc26f25)

``` python

y = df.iloc[:,1].values
y

```
![image](https://github.com/user-attachments/assets/39e61263-1526-4266-8256-09d3e50777bf)

``` python

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

```
``` python

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

```
``` python
y_pred

```
![image](https://github.com/user-attachments/assets/3cba9609-0114-4c24-9256-0a3355b65260)

``` python

y_test

```
![image](https://github.com/user-attachments/assets/f7060532-62f4-49c9-bc7d-5fb537b16db0)

```python

mse = mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae = mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse = np.sqrt(mse)
print("RMSE = ",rmse)

```
![image](https://github.com/user-attachments/assets/f30f7919-3095-41e0-99fb-d3890e705ce4)

``` python

plt.scatter(x_train,y_train,color="black")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```
![image](https://github.com/user-attachments/assets/744dd1f3-6d4f-4c7e-b71a-bf0174a1c41d)

``` python

plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,y_pred,color="black")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```
![image](https://github.com/user-attachments/assets/4261178e-289c-44c3-89a4-1442e878fd7c)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
