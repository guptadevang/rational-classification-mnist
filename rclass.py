import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Function definition
def f(params, x, y):
    a, b, c, d, e = params
    residuals = y - (a * x**2 + b * x + c) / (d * x + e)
    return np.sum(residuals**10)

# Generating synthetic data
np.random.seed(0)
x = np.linspace(1, 10, 100)
y = 3*x**2 + 2*x + 1 + np.random.normal(0, 10, size=100)
print("X:", x)
print("Y:", y)

# Initial parameters guess with random values
np.random.seed()
initial_params = np.random.rand(5)  # Generates five random numbers between 0 and 1
print("Initial random parameters:", initial_params)

# Optimization using BFGS
result = minimize(f, initial_params, args=(x, y), method='BFGS')

# Extract optimized parameters
optimized_params = result.x
print("Optimized parameters:", optimized_params)

plt.figure(figsize=(6, 5))
plt.scatter(x, y, label='Data points')
# Calculate the fitted curve using the optimized parameters
fitted_y = (optimized_params[0] * x**2 + optimized_params[1] * x + optimized_params[2]) / (optimized_params[3] * x + optimized_params[4])
plt.plot(x, fitted_y, color='red', label='Fitted curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Data and Fitted Curve')
plt.tight_layout()
plt.savefig('optimized_fitted_curve.png')
plt.show()