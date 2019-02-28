import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


#Specifying size of the figure.
plt.rcParams['figure.figsize'] = (20.0,10.0)


#Reading data from a csv file using Pandas module.
data = pd.read_csv("Fuel-Efficiency-Data.csv")


#Fetching the values from the csv file to create an object of Independent and Dependent variable.
X = data['Miles'].values
Y = data['Gas'].values


#Finding the mean of both the variables.
mean_x = np.mean(X)
mean_y = np.mean(Y)

#Calculating the no. of elements present.
n = len(X)


#Process to find the slope of the regression line using the formula M = Sum of ((X - Xmean)*(Y - Ymean)) / Sum of (X - Xmean)^2
numerator = 0
denominator = 0

for i in range(n):
    numerator = numerator + (X[i]-mean_x)*(Y[i]-mean_y)
    denominator = denominator + (X[i] - mean_x) ** 2

m = numerator/denominator

#Calculating Y Intercept of the line using formula c = Y - mX
c = mean_y - (m * mean_x)

Rnum = 0
Rden = 0
yp = []


#Calculating the R-Squared value or the Coefficient of Determination that shows the goodness of the fit of the regression line.
for i in range(len(X)):
    yp.append(m*X[i]+c)
    Rnum += (yp[i] - mean_y) ** 2
    Rden += (Y[i] - mean_y) ** 2

Rsquare = Rnum/Rden


#Predicting the gallons for the 1200 miles of travel.
DepPrediction = m*1200 + c
print("Value of Co-efficient of determination is:",Rsquare)
print("Value predicted through the model for 1200 miles is: %d Gallons (Approximately)" % int(DepPrediction))

#Plotting the values on a graph

min_x = np.min(X)
max_x = np.max(X)

x = np.linspace(min_x,max_x,1000)
y = m*x + c


plt.plot(x,y,label="Regression Line")
plt.scatter(X,Y,label="Scatter Plot of Variables")


plt.xlabel("Miles Travelled")
plt.ylabel("Gas Required")

plt.legend()
plt.show()

#Using the Sklearn module to make prediction
X = X.reshape(-1,1)
Xnew = [[1200]]

reg = LinearRegression()
reg.fit(X,Y)
Ypredicted = reg.predict(Xnew)

print("Value predicted through the Sklearn module for 1200 miles is: %d Gallons (Approximately)" % int(Ypredicted))


