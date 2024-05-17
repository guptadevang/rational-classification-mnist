import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Data generation function
def generate_data(num_points, noise_level=1.0):
    # Generating x-values
    x = np.linspace(-2, 2, num_points)
    
    # Generating y-values
    y = np.abs(x)
    y[x > 0] += 2  # Adding a step discontinuity at x = 0
    
    # Adding Gaussian noise
    y_noisy = y + np.random.normal(0, noise_level, size=x.shape)
    
    return x, y_noisy

# Rational function for fitting
def rational(params, x, y):
    a, b, c, d, e = params
    model = (a * x**2 + b * x + c) / (d * x + e)
    residuals = y - model
    return np.sum(residuals**10)

# Generate data
x, y = generate_data(10, noise_level=0.01)

# Generate random initial parameters
#initial_params = np.random.normal(0, 1, 5) 
initial_params = np.array([1, 1, 1, 1, 1])

# Rational fitting
result = minimize(rational, initial_params, args=(x, y), method='BFGS')
optimized_params = result.x
rational_y = (optimized_params[0] * x**2 + optimized_params[1] * x + optimized_params[2]) / (optimized_params[3] * x + optimized_params[4])

# Polynomial fitting
polynomial_coeffs = np.polyfit(x, y, deg=5)
poly_y = np.polyval(polynomial_coeffs, x)

# Formatting labels
poly_params_label = 'Poly Coeffs: ' + ', '.join(f'{coeff:.2f}' for coeff in polynomial_coeffs)
rational_params_label = f'Rational Coeffs: a={optimized_params[0]:.2f}, b={optimized_params[1]:.2f}, c={optimized_params[2]:.2f}, d={optimized_params[3]:.2f}, e={optimized_params[4]:.2f}'

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Noisy data')
plt.plot(x, np.abs(x) + (x > 0) * 2, 'k--', label='True function')
plt.plot(x, poly_y, color='green', label=f'Polynomial fit (deg=5) | {poly_params_label}')
plt.plot(x, rational_y, color='orange', label=f'Rational fit | {rational_params_label}')
plt.legend()
plt.title('Comparing Model Fits to Non-smooth Data with BFGS Optimization')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()