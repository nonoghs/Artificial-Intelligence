import sys
import csv
import matplotlib.pyplot as plt
import numpy as np


def load_and_visualize(filename):
    """Load data from CSV and visualize it."""
    years, days = [], []
    with open(filename, 'r') as file:
        data = csv.reader(file)
        next(data)  # Skip header
        for row in data:
            years.append(int(row[0]))
            days.append(int(row[1]))

    plt.plot(years, days)
    plt.xticks(years)
    plt.xlabel('Years')
    plt.ylabel('Days')
    plt.savefig("plot.jpg")
    return years, days


def construct_matrices(x_vals, y_vals):
    """Construct matrices for computation."""
    X = np.vstack([np.ones(len(x_vals)), x_vals]).T
    print("Q3a:")
    print(X)

    Y = np.array(y_vals).reshape(-1, 1)
    print("Q3b:")
    print(Y)

    return X, Y


def compute_parameters(X, Y):
    """Compute parameters using the normal equation."""
    XT_X = np.dot(X.T, X)
    print("Q3c:")
    print(XT_X)

    XT_X_inv = np.linalg.inv(XT_X)
    print("Q3d:")
    print(XT_X_inv)

    pseudo_inverse = np.dot(XT_X_inv, X.T)
    print("Q3e:")
    print(pseudo_inverse)

    params = np.dot(pseudo_inverse, Y)
    print("Q3f:")
    print(params)

    return params


def analyze_prediction(params):
    """Analyze and print the prediction."""
    pred_value = params[0] + params[1] * 2022
    print(f"Q4: {pred_value[0]}")

    trend_direction = ">" if params[1] > 0 else "<" if params[1] < 0 else "="
    print(f"Q5a: {trend_direction}")

    trend = "increases" if params[1] > 0 else "decreases" if params[1] < 0 else "remains constant"
    print(f"Q5b: This means predict y {trend} when input x increases.")

    print(f"Q6a: {-params[0][0] / params[1][0]}")
    print(
        "Q6b: Based on the model, the trend suggests a significant change over the years. However, predictions should be made with caution due to potential data variations.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <filename>")
        sys.exit(1)

    x, y = load_and_visualize(sys.argv[1])
    X, Y = construct_matrices(x, y)
    parameters = compute_parameters(X, Y)
    analyze_prediction(parameters)