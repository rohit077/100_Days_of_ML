#Importing Libraries
import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
import utils

#Loading the data
data = np.loadtxt(os.path.join('Data', 'ex2data1.txt'), delimiter=',')
X, y = data[:, 0:2], data[:, 2]

#Visualising the data
def plotdata(X, y):
    fig = pyplot.figure()
    pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)

plotData(X, y)

#Defining the Logistic Sigmoid Function 𝜎(𝑧)
def sigmoid(z):
    return 1/(1 + np.exp(-z))

#Cost and Gradient at Initialization
A = sigmoid(np.dot(w.T, X) + b)
cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))

#Implement Gradient Descent from Scratch


#Plotting the Convergence of 𝐽(𝜃)


#Plotting the Decision Boundary


#Predictions Using the Optimized 𝜃 Values
