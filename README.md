# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program and import the required libraries such as NumPy, Matplotlib, and Linear Regression from sklearn.
2. Define the dataset by storing the hours studied as the independent variable (X) and marks scored as the dependent variable (Y). 
3. Create the Simple Linear Regression model and train it using the given dataset.
4. Predict the marks using the trained regression model for the given input values.
5. Plot the graph by displaying the actual data points using a scatter plot and the regression line using a line plot, then display the graph.

## Program:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by:Yeshwanth.S
RegisterNumber:212225060310
```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data={
    "Hours_Studied":[1,2,3,4,5,6,7,8,9,10],
    "Marks_Scored":[35,40,50,55,60,65,70,80,85,95]
}
df=pd.DataFrame(data)
print("Dataset:\n",df.head())
df
x=df[["Hours_Studied"]]
y=df["Marks_Scored"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("\nModel Parameters:")
print("Intercept (b0):",model.intercept_)
print("Slope (b1):",model.coef_[0])

print("\nEvaluation Metrics:")
print("Mean Squared Error :",mean_squared_error(y_test,y_pred))
print("R-squared :",r2_score(y_test,y_pred))
plt.figure(figsize=(8,6))
plt.scatter(x,y,color="blue",label="Actual Data")
plt.plot(x,model.predict(x),color="red",linewidth=2,label="Regression Line")
plt.xlabel("Hours_Studied")
plt.ylabel("Marks_Scored")
plt.title("Simple Linear Regression : Prediction Marks")
plt.grid(True)
plt.legend()
plt.show()
```


## Output:
<img width="886" height="1026" alt="image" src="https://github.com/user-attachments/assets/5b960e26-cb18-4c26-b967-7c23daa3928a" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
