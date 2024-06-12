#Importing Libraries
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import pandas as pd
import seaborn as sns
plt.rcParams['figure.figsize'] = (12, 8)

#Loading the data
data = pd.read_csv('bike_sharing_data.txt')
data.head()
data.info()

#Visualising the Data
ax = sns.scatterplot(x = " Total Population", y = " Total Profit", data = data)
ax.set_title = ("Total Profit in $10,000 vs. City population in 10,000's")

#Cost Function
def Cost(X, y, theta):
    m = len(y)
    y_pred = X.dot(theta)
    result = 1 / (2 * m) * np.sum((y_pred - y) ** 2)

    return result

#Calculating the Cost
m = data.population.values.size
X = np.append(np.ones((m, 1)), data.population.values.reshape(m, 1), axis = 1)
y = data.Profit.values.reshape(m, 1)
theta = zeros((2, 1))

Cost(X, y, theta)


#Gradient descent
def Grad(X, y, theta, alpha, iterations):
    m = len(y)
    costs = []
    for i in range(iterations):
        y_pred = X.dot(theta)
        result = 1 / (2 * m) * np.sum((y_pred - y) ** 2)
        theta = theta - (alpha / m) * (np.dot((X.transpose()), (y_pred - y)))
