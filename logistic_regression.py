import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('train.csv')
df = df[['Survived','Name', 'Pclass', 'Age', 'Sex', 'Fare', 'Parch','SibSp']]
df['Sex'].replace('female',0,inplace=True)
df['Sex'].replace('male',1,inplace=True)
df.fillna(df.mean(), inplace=True)
y = df['Survived']
X = df[['Pclass', 'Age', 'Sex', 'Fare', 'Parch','SibSp']]

# Feature scalling of X
X = (X - X.mean(axis=0))/X.std(axis=0)
m = np.size(y)
X.insert(loc=0, column='ones', value=np.ones(m,dtype=int))
X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0)
theta = np.zeros(7,dtype = int)
alpha = 0.01 # Learning rate
num_iters = 300 # number of iterations 
m = np.size(y_train)


def computeCost(X_train,y_train,theta):
    h = 1/(1+np.exp(-(np.dot(X_train,theta))))
    y1 = 1-y_train
    J = (-1/m)*((y_train.T).dot(np.log(h)) + (y1.T).dot(np.log(1-h)))
    return J


def gradientDescent(X_train, y_train, theta, alpha, num_iters):
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        h = 1/(1+np.exp(-(np.dot(X_train,theta))))
        theta = theta - (alpha/m)*(X_train.T.dot(np.subtract(h,y_train)))
        J_history[i] = computeCost(X_train, y_train, theta)
    return theta, J_history

def predict(X_test):
    predictions = np.ones(np.size(y_test),dtype = int)
    hpredict = 1/(1+np.exp(-(np.dot(X_test,theta))))
    predictions[hpredict < 0.5 ] = 0
    return predictions


theta,J_history = gradientDescent(X_train, y_train, theta, alpha, num_iters)
predictions = predict(X_test)

print('Accuracy Score: {}'.format(accuracy_score(y_test,predictions)))
