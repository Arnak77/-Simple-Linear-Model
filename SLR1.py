# import require library 
import numpy as np 	

import matplotlib.pyplot as plt

import pandas as pd	

# import the dataset
dataset = pd.read_csv(r'D:\NIT\DECEMBER\11 DEC  (SLR(SIMPLE))\11th - Regression model\SIMPLE LINEAR REGRESSION\Salary_Data.csv')

# split the data to independent variable 
X = dataset.iloc[:,:-1].values
# split the data to dependent variabel 
y = dataset.iloc[:,1].values 

# as d.v is continus that regression algorithm 
# as in the data set we have 2 attribute we slr algo


# split the dataset to 70-30%
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

#we called simple linear regression algoriytm from sklearm framework 
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

# we build simple linear regression model regressor
regressor.fit(X_train, y_train)

# test the model & create a predicted table 
y_pred = regressor.predict(X_test)



plt.scatter(X_test, y_test, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary  vs Experience')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_train, y_train, color = 'red') 
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# slope is generrated from linear regress algorith which fit to dataset 
m = regressor.coef_
m

# interceppt also generatre by model. 
c = regressor.intercept_
c

# predict or forcast the future the data which we not trained before 

y_16=9312 * 16 + 26780


