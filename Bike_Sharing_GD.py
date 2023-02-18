import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('hour.csv')

##normalization hr
#data.iloc[:,5] = (data.iloc[:,5] - data.iloc[:,5].min())/(data.iloc[:,5].max() - data.iloc[:,5].min())
##normalization cnt
#data.iloc[:,16] = (data.iloc[:,16] - data.iloc[:,16].min())/(data.iloc[:,16].max() - data.iloc[:,16].min())
##print(data.iloc[:,5])

data.iloc[:,1] = pd.to_datetime(data.iloc[:,1], errors='coerce')
data.iloc[:,1] = (data.iloc[:,1].dt.day)

x = data.iloc[:,1:14]
y = data.iloc[:,16]

#print(x.dtypes);
X_train, X_test , Y_train , Y_test = train_test_split(x , y, test_size = 0.2)



X__train = np.c_[np.ones((len(X_train),1)),X_train]
X__test = np.c_[np.ones((len(X_test),1)),X_test]
#print(X__train)
theta = np.random.random(X__train.shape[1])
#theta = np.zeros(X__train.shape[1])

def calCost(theta,X,Y):
    m = len(Y)

    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions - Y))
    return cost


def gradientDescent(X, Y, theta , alpha = 0.00001, iterations = 10000000):

    m = len(Y)
    costHistory = np.zeros(iterations)
    thetaHistory = np.zeros((iterations , X__train.shape[1]))
    
    for it in range(iterations):

        prediction = np.dot(X,theta)

        theta = theta - (1/(2*m)) * alpha * (X.T.dot((prediction - Y)))
        thetaHistory[it, :] = theta.T
        costHistory = calCost(theta,X,Y)
        
        return theta, costHistory , thetaHistory

theta , costHistory , thetaHistory = gradientDescent(X__train,Y_train,theta) 

print(costHistory , theta)

m = len(Y_test)


#error
for i in range(m):
    error = calCost(theta,X__test,Y_test)
   
print(error)





   











#def gradientDescent(X, Y, theta, alpha, m, numIterations):
#    xTrans = x.transpose()
#    for i in range(0, numIterations):
#        hypothesis = np.dot(x, theta)
#        loss = hypothesis - y 
#        cost = np.sum(loss ** 2) / (2 * m)
#        print("Iteration %d | Cost: %f" % (i, cost))
#        # avg gradient per example
#        gradient = np.dot(xTrans, loss) / m 
#        # update
#        theta = theta - alpha * gradient
#    return theta







