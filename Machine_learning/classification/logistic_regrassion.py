import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class LogisticRegression:
    def __init__(self, learning_rate=0.01):
        self.weights = None
        self.lr = learning_rate
        self.losses = []
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        normalized_pred = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return normalized_pred 

    def fit(self, train_x, train_y, iteration=1000):

        m, n = train_x.shape
        k = len(np.unique(train_y))

        self.weights = np.zeros((n+1, k))
        X = np.c_[np.ones(m), train_x] 
        Y = np.eye(k)[train_y]

        for _ in range(iteration):
            z = np.dot(X, self.weights)
            y_pred = self.softmax(z)

            loss = (-1/m) * np.sum(Y * np.log(y_pred + 1e-15))
            self.losses.append(loss)

            gradient = (1/m) * np.dot(X.T, (y_pred - Y))

            self.weights -= self.lr * gradient


    def predict(self, test_x):
        m, n = test_x.shape
        X = np.c_[np.ones(m), test_x]
        Z = np.dot(X, self.weights)
        Y = self.softmax(Z)

        print(Y)
        return np.argmax(Y, axis=1)


if __name__ == '__main__':
   
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train softmax regression
    model = LogisticRegression(learning_rate=0.1)
    model.fit(X_train, y_train, iteration=1000)

    # Predictions
    y_pred = model.predict(X_test)

    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy) 