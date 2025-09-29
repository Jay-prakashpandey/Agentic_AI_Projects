import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression as SklearnLinearRegression


class LinearRegression:
    def __init__(self, penalty=None, alpha=0.0, l1_ratio=0.5):
        """
        penalty: None | 'ridge' | 'lasso' | 'elasticnet'
        alpha: regularization strength
        l1_ratio: balance between L1 and L2 (only for elasticnet)
        """
        self.parameters = {}  # Stores model parameters: slope (m) and intercept (c)
        self.penalty = penalty  # Type of regularization
        self.alpha = alpha      # Regularization strength
        self.l1_ratio = l1_ratio  # ElasticNet mixing parameter

    def save(self, filepath):
        """Save model parameters to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.parameters, f)

    def load(self, filepath):
        """Load model parameters from a JSON file."""
        with open(filepath, 'r') as f:
            self.parameters = json.load(f)

    def forward_propagation(self, train_input):
        """
        Predict output using current parameters.
        Formula: y_pred = m * x + c
        """
        m = self.parameters['m']
        c = self.parameters['c']
        predictions = np.multiply(m, train_input) + c
        return predictions

    def cost_function(self, predictions, train_output):
        """
        Compute Mean Squared Error (MSE) and add regularization if specified.
        MSE: (1/n) * sum((y_true - y_pred)^2)
        Ridge: MSE + alpha * m^2
        Lasso: MSE + alpha * |m|
        ElasticNet: MSE + alpha * (l1_ratio * |m| + (1-l1_ratio) * m^2)
        """
        mse = np.mean((train_output - predictions) ** 2)
        m = self.parameters['m']

        # Add regularization depending on penalty
        if self.penalty == "ridge":
            cost = mse + self.alpha * (m ** 2)
        elif self.penalty == "lasso":
            cost = mse + self.alpha * np.abs(m)
        elif self.penalty == "elasticnet":
            cost = mse + self.alpha * (self.l1_ratio * np.abs(m) +
                                       (1 - self.l1_ratio) * (m ** 2))
        else:
            cost = mse
        return cost

    def backward_propagation(self, train_input, train_output, predictions):
        """
        Compute gradients (derivatives) for parameters m and c.
        dM = 2 * mean(x * (y_pred - y_true)) + regularization
        dC = 2 * mean(y_pred - y_true)
        """
        derivatives = {}
        df = (predictions - train_output)  # Error term
        dm = 2 * np.mean(np.multiply(train_input, df))  # Gradient w.r.t m
        dc = 2 * np.mean(df)                            # Gradient w.r.t c

        m = self.parameters['m']

        # Add regularization derivatives
        if self.penalty == "ridge":
            dm += 2 * self.alpha * m  # Ridge: derivative of alpha * m^2 is 2 * alpha * m
        elif self.penalty == "lasso":
            dm += self.alpha * np.sign(m)  # Lasso: derivative of alpha * |m| is alpha * sign(m)
        elif self.penalty == "elasticnet":
            dm += self.alpha * (self.l1_ratio * np.sign(m) +
                                2 * (1 - self.l1_ratio) * m)  # ElasticNet: combination

        derivatives['dm'] = dm
        derivatives['dc'] = dc
        return derivatives

    def update_parameters(self, derivatives, learning_rate):
        """
        Update parameters using gradients and learning rate.
        m = m - lr * dm
        c = c - lr * dc
        """
        self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']

    def train(self, train_input, train_output, learning_rate, iters):
        """
        Train the model using gradient descent.
        - Randomly initialize m and c
        - For each iteration:
            - Predict outputs
            - Compute loss
            - Compute gradients
            - Update parameters
        """
        self.parameters['m'] = np.random.uniform(-0.1, 0.1)
        self.parameters['c'] = np.random.uniform(-0.1, 0.1)
        self.loss = []
        for i in range(iters):
            predictions = self.forward_propagation(train_input)
            cost = self.cost_function(predictions, train_output)
            derivatives = self.backward_propagation(train_input, train_output, predictions)
            self.update_parameters(derivatives, learning_rate)
            self.loss.append(cost)
            print(f"Iteration = {i + 1}, Loss = {cost:.4f}")
        return self.parameters, self.loss

    def train_with_animation(self, train_input, train_output, learning_rate, iters):
        """
        Same as train(), but animates the regression line updating over iterations.
        Saves animation as a GIF.
        """
        self.parameters['m'] = np.random.uniform(-0.1, 0.1) 
        self.parameters['c'] = np.random.uniform(-0.1, 0.1)
        self.loss = []

        fig, ax = plt.subplots()
        x_vals = np.linspace(min(train_input), max(train_input), 100)
        line, = ax.plot(x_vals, self.parameters['m'] * x_vals + self.parameters['c'],
                        color='red', label='Regression Line')
        ax.scatter(train_input, train_output, marker='o', color='green', label='Training Data')

        ax.set_ylim(0, max(train_output) + 1)

        def update(frame):
            predictions = self.forward_propagation(train_input)
            cost = self.cost_function(predictions, train_output)
            derivatives = self.backward_propagation(train_input, train_output, predictions)
            self.update_parameters(derivatives, learning_rate)
            line.set_ydata(self.parameters['m'] * x_vals + self.parameters['c'])
            self.loss.append(cost)
            print(f"Iteration = {frame + 1}, Loss = {cost:.4f}")
            return line,

        ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)
        ani.save('linear_regression_A.gif', writer='ffmpeg')

        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title(f'Linear Regression ({self.penalty if self.penalty else "plain"})')
        plt.legend()
        plt.show()

        return self.parameters, self.loss
     
    def predict(self, test_input):
        """
        Predict output for given input X using the trained model.
        Formula: y_pred = m * X + c
        """
        return self.forward_propagation(test_input)

if __name__ == "__main__":
    # Generate synthetic data for training
    # np.random.seed(0)
    # X = 2 * np.random.rand(100)  # 100 random values between 0 and 2
    # y = 4 + 3 * X + np.random.randn(100)  # y = 4 + 3x + noise

    # step-1 : Load data from CSV
    url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240320114716/data_for_lr.csv'
    data = pd.read_csv(url)
    # step-2 : Preprocess data (handle missing values)
    data = data.dropna()
    # step-3 : Extract training input and output variables
    train_input = np.array(data.x[0:500]).reshape(500, 1)
    train_output = np.array(data.y[0:500]).reshape(500, 1)
    # step-4 : Prepare data for model testing
    test_input = np.array(data.x[500:700]).reshape(199, 1)
    test_output = np.array(data.y[500:700]).reshape(199, 1)

    # 1. Plain Linear Regression (default MSE)
    model = LinearRegression()
    params, loss = model.train(train_input=train_input, train_output=train_output, learning_rate=0.01, iters=50)
    # 2. Ridge Regression (uncomment to use)
    # model = LinearRegression(penalty='ridge', alpha=0.1)
    # model = sklearn.linear_model.Ridge(alpha=0.1)
    # parameters, loss = model.train(X, y, learning_rate=0.01, iters=100)
    # 3. Lasso Regression (uncomment to use)
    # model = LinearRegression(penalty="lasso", alpha=0.1)
    # model = sklearn.linear_model.Lasso(alpha=0.1)
    # params, loss = model.train(X, y, learning_rate=0.01, iters=50)
    # 4. Elastic Net (uncomment to use)
    # model = LinearRegression(penalty="elasticnet", alpha=0.1, l1_ratio=0.7)
    # model = sklearn.linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7)
    # params, loss = model.train(X, y, learning_rate=0.01, iters=50)

    # --- Model Evaluation on Test Data ---
    # Predict outputs for the test set using the trained model
    y_pred = model.predict(test_input)

    # Calculate Mean Squared Error: average of squared differences between actual and predicted values
    mse = mean_squared_error(test_output, y_pred)

    # Calculate Root Mean Squared Error: square root of MSE, gives error in original units
    rmse = np.sqrt(mse)

    # Calculate Mean Absolute Error: average of absolute differences between actual and predicted values
    mae = mean_absolute_error(test_output, y_pred)

    # Calculate R^2 Score: proportion of variance in the dependent variable explained by the model (1 is perfect)
    r2 = r2_score(test_output, y_pred)

    # Print evaluation metrics
    # print("\nModel Evaluation on Test Data:")
    # print(f"Mean Squared Error (MSE): {mse:.4f}")
    # print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    # print(f"Mean Absolute Error (MAE): {mae:.4f}")
    # print(f"R^2 Score: {r2:.4f}")

    # print("Trained parameters:", params)
    # print("Loss over iterations:", loss)

    # --- scikit-learn's LinearRegression ---
    sk_model = SklearnLinearRegression()
    sk_model.fit(train_input, train_output)
    y_pred_sklearn = sk_model.predict(test_input)

    mse_sk = mean_squared_error(test_output, y_pred_sklearn)
    mae_sk = mean_absolute_error(test_output, y_pred_sklearn)
    r2_sk = r2_score(test_output, y_pred_sklearn)

    print("\nscikit-learn LinearRegression Evaluation:")
    print(f"MSE_sk: {mse_sk:.4f}, my_rmse_sk: {np.sqrt(mse_sk):.4f}")
    print(f"MAE_sk: {mae_sk:.4f}, my_mae_sk: {mae_sk:.4f}")
    print(f"R^2_sk: {r2_sk:.4f}, my_r2_sk: {r2_sk:.4f}")
