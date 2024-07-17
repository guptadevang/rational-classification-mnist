import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib as mpl

# Configure Matplotlib to use xelatex
mpl.rcParams.update({
    "pgf.texsystem": "xelatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
})

# Define the piecewise function
def piecewise_function(x):
    return np.piecewise(x, 
                        [x < -5, (x >= -5) & (x <= 2), x > 2], 
                        [3, -4, 1])

# Generate points
total_points = 100
x_points = np.linspace(-10, 10, total_points)
y_points = piecewise_function(x_points)

# Define the rational function
def Rational_function(x, p0, p1, p2, p3, p4, p5, q0):
    return (p0 + p1*x + p2*x**2 + p3*x**3 + p4*x**4 + p5*x**5) / (q0 * (x + 5) * (x - 2))

# Initial guess for optimization
initial_guess = np.random.rand(7) * 0.1

# Objective function
def objective_function(params):
    return np.sum((y_points - Rational_function(x_points, *params))**2)

# BFGS optimization
result_bfgs = minimize(objective_function, initial_guess, method='BFGS')
params_bfgs = result_bfgs.x

# Bisection method
def bisection_method(objective_func, bounds, tol=1e-5):
    params = np.zeros(len(bounds))
    for i, (low, high) in enumerate(bounds):
        while (high - low) / 2.0 > tol:
            mid = (low + high) / 2.0
            params[i] = mid
            if objective_func(params) * objective_func([params[j] if j != i else low for j in range(len(params))]) < 0:
                high = mid
            else:
                low = mid
        params[i] = (low + high) / 2.0
    return params

bounds = [(-1, 1)] * 7
params_bisect = bisection_method(objective_function, bounds)

# Plot results
x = np.linspace(-10, 10, 1000)
y = piecewise_function(x)
y_fit_bfgs = Rational_function(x, *params_bfgs)
y_fit_bisect = Rational_function(x, *params_bisect)

# Plot BFGS fit
plt.figure(figsize=(6, 3.6))
plt.plot(x, y, label='Piecewise Function')
plt.plot(x, y_fit_bfgs, label='Fitted Rational Function (BFGS)', linestyle=':')
plt.scatter(x_points, y_points, color='red', s=10, label='Equidistant points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Piecewise Function with Fitted Rational Function (BFGS)')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/one_plot_bfgs.pgf')
plt.show()

# Plot Bisection fit
# Plot Bisection fit
plt.figure(figsize=(6, 3.6))
plt.plot(x, y, label='Piecewise Function')
plt.plot(x, y_fit_bisect, label='Fitted Rational Function (Bisection)', linestyle='-.')
plt.scatter(x_points, y_points, color='red', s=10, label='Equidistant points')

# Set the limits for the x and y axes
plt.xlim([-7.5, 5])
plt.ylim([-20000, 20000])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Piecewise Function with Fitted Rational Function (Bisection)')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/one_plot_bisection.pgf')
plt.show()

# Calculate distances
distances_bfgs = np.abs(y_points - Rational_function(x_points, *params_bfgs))
distances_bisect = np.abs(y_points - Rational_function(x_points, *params_bisect))

print("\nNumber of iterations (BFGS):", result_bfgs.nit)
print("Distances from each point to the curve (BFGS):")
print(distances_bfgs)
print("Coefficients (BFGS):", params_bfgs)

print("\nNumber of iterations (Bisection):", len(params_bisect))
print("Distances from each point to the curve (Bisection):")
print(distances_bisect)
print("Coefficients (Bisection):", params_bisect)