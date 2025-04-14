# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
``` python
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PREETHI D 
RegisterNumber: 212224040250


import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Downloads\student_scores.csv')

print(df.head())
print(df.tail())

#data to variables
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#predicted values 
print(y_pred)

#actual values
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
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
```

## Output:
``` python
df.head()
```
![image](https://github.com/user-attachments/assets/9faa4c9c-d281-4027-a488-3b2a9a7d9e3a)
``` python
df.tail()
```
![image](https://github.com/user-attachments/assets/4b7bd42d-7031-4a0a-86fb-a9bfce1e286d)

``` python
X Array Value
```
![image](https://github.com/user-attachments/assets/60e98cff-df52-4438-88ff-8eb5ae7409d9)

```python
Y Array value
```
![image](https://github.com/user-attachments/assets/7465f9ae-85ae-483e-89eb-c97f85181d61)

``` python
Values of Y prediction
```
![image](https://github.com/user-attachments/assets/cbb3a22c-e2ec-4dbc-b506-ff166294a18c)

``` python
Values of Y Test
```
![image](https://github.com/user-attachments/assets/bfbaba54-b251-42e1-86f6-31c36fc738bc)

``` python
Training Set graph
```
![image](https://github.com/user-attachments/assets/27bb7eef-e0de-475e-8b74-eaf6924c934a)

``` python
Test Set graph
```
![image](https://github.com/user-attachments/assets/ef56a89e-f9c4-42d1-92f5-de7411f64f7b)

``` python
Values of MSE, MAE and RMSE:
```
![image](https://github.com/user-attachments/assets/f0675338-5699-4769-b7f2-1cf3fc3c963b)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
