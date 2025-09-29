import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        

class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.weights = None
        self.losses = []
        
    def fit(self, train_x, train_y, iterations=50):
        n_samples, n_features = train_x.shape
        self.weights = np.zeros(n_features+1)
        X = np.c_[np.ones(n_samples), train_x]  # Add bias term
        train_y = train_y.reshape(-1)   # or train_y.ravel()
        for _ in range(iterations):
            y_predicted = np.dot(X, self.weights)
            dw = (1/n_samples) * np.dot(X.T, (y_predicted-train_y))
            self.weights -= self.lr * dw 
            
            # track loss
            loss = np.mean((y_predicted - train_y) ** 2)
            self.losses.append(loss)

            # safety check (avoid nan explosion)
            if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
                print("⚠️ Weights diverged, try reducing learning rate.")
                break

    def predict(self, test_x):
        m, n = test_x.shape
        X = np.c_[np.ones(m), test_x]
        y_pred = np.dot(X, self.weights)
        return y_pred

if __name__ == '__main__':
    # step-1 : Load data from CSV
    url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240320114716/data_for_lr.csv'
    data = pd.read_csv(url).dropna()

    print(data.info)

    # Normalize data
    x_mean, x_std = data.x.mean(), data.x.std()
    data.x = (data.x - x_mean) / x_std

    # Train/Test split
    train_input = np.array(data.x[0:500]).reshape(500, 1)
    train_output = np.array(data.y[0:500]).reshape(500, 1)
    test_input = np.array(data.x[500:699]).reshape(-1, 1)  # will automatically use 199 x 1
    test_output = np.array(data.y[500:699]).reshape(-1, 1)

    model = LinearRegression(learning_rate=0.01)
    model.fit(train_input, train_output, iterations=1000)

    y_pred = model.predict(test_input)

    # Metrics
    mse = mean_squared_error(test_output, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_output, y_pred)
    r2 = r2_score(test_output, y_pred)

    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # Plot training loss
    plt.plot(model.losses)
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.show()