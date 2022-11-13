# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.
2.Assign hours to X and scores to Y.
3.Implement training set and test set of the dataframe. 
4.Plot the required graph both for test data and training data and Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by  : PRASANTH E
RegisterNumber: 22007885 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
df.tail()
#segregation data to variables
X=df.iloc[:,:-1].values
X
Y=df.iloc[:,1].values
Y
#spliting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted value
Y_pred
#displaying actual value
Y_test
#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_train,regressor.predict(X_train),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(Y_test,Y_pred)
print('MSC=',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE=',mae)

rmse=np.sqrt(mse)
print("RMSE=",rmse)

```

## Output:

![image](https://user-images.githubusercontent.com/114572171/201523084-e81ddd2f-5eea-4383-9c61-32e4c68f4f54.png)

![image](https://user-images.githubusercontent.com/114572171/201523098-affbe9a9-3bf7-4ae6-937e-0d495486f419.png)

![image](https://user-images.githubusercontent.com/114572171/201523107-3ee002bd-f069-4df4-9daf-008d9d9a0166.png)

![image](https://user-images.githubusercontent.com/114572171/201523117-f4885995-aad1-4589-b8d6-88453caf2926.png)

![image](https://user-images.githubusercontent.com/114572171/201523124-5bb5b6eb-7b53-446f-98cf-157df3ad019b.png)

![image](https://user-images.githubusercontent.com/114572171/201523587-2c1891af-cb6a-4d0c-a8ad-5d1d54ccdca1.png)

![image](https://user-images.githubusercontent.com/114572171/201523339-42a32abb-2bac-4a6c-a2d5-77105d607d88.png)

![image](https://user-images.githubusercontent.com/114572171/201523348-3ee7f4c3-ebe0-4069-801e-8f105c53c958.png)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
