import sympy as sp
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.optimize import minimize

# Configure Matplotlib to use xelatex
mpl.rcParams.update({
    "pgf.texsystem": "xelatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
})

# Function defining our piecewise function f(x)
def f(x):
    x1, x2 = x
    return np.where((x1 + x2 < 1), 0, 1)

# Created a grid of points in the range [0, 1] for both x1 and x2
grid_size = 10 
x1_points, x2_points = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
x1_points = x1_points.flatten()
x2_points = x2_points.flatten()
x_points = np.vstack((x1_points, x2_points))

# It calculates y values using the function f(x)
y_points = f(x_points)

def generate_indices(alpha, n):
    indices = []
    for comb in product(range(alpha + 1), repeat=n):
        if sum(comb) <= alpha:
            indices.append(comb)
    return indices

# Generates a rational function r(x) = p(x)/q(x)
def generate_rational_function(alpha_num, alpha_den, n):
    variables = sp.symbols('x1:' + str(n + 1))
    p_indices = generate_indices(alpha_num, n)
    q_indices = generate_indices(alpha_den, n)

    numerator = 0
    for idx in p_indices:
        term = 1
        for i, exp in enumerate(idx):
            term *= variables[i]**exp
        idx_str = ''.join(map(str, idx))
        numerator += sp.symbols(f'p{idx_str}') * term

    denominator = 0
    for idx in q_indices:
        term = 1
        for i, exp in enumerate(idx):
            term *= variables[i]**exp
        idx_str = ''.join(map(str, idx))
        denominator += sp.symbols(f'q{idx_str}') * term

    rational_function = numerator / denominator
    return rational_function, p_indices, q_indices, variables

# Adjust degrees and order for numerator and denominator of the rational function
alpha_num = 5
alpha_den = 4
n = 2

rational_function, p_indices, q_indices, variables = generate_rational_function(alpha_num, alpha_den, n)
print("Generated rational function:")
print(rational_function)

# Defined the difference function to minimize (r(x) - f(x))^2
def difference_function(x, params):
    x1, x2 = x
    p = params[:len(p_indices)]
    q = params[len(p_indices):]
    
    num = sum([p[i] * (x1**idx[0]) * (x2**idx[1]) for i, idx in enumerate(p_indices)])
    den = sum([q[i] * (x1**idx[0]) * (x2**idx[1]) for i, idx in enumerate(q_indices)])
    
    rational_val = num / den
    actual_val = f((x1, x2))
    difference = (rational_val - actual_val)**10
    
    return rational_val, difference


def total_difference(params, x_points, y_points):
    total_diff = 0
    for i in range(x_points.shape[1]):
        _, diff = difference_function((x_points[0, i], x_points[1, i]), params)
        total_diff += diff
    
    return total_diff

def generate_initial_params(p_indices, q_indices):
    initial_params = [1] * len(p_indices) + [0.5] * len(q_indices)
    return np.array(initial_params)



# Optimized parameters using the bisection method
def coordinate_bisection(params, x_points, y_points, tol=1e-15, max_iter=10000, convergence_threshold=1e-6):
    start_time = time.time()
    n_params = len(params)
    previous_total_diff = total_difference(params, x_points, y_points)
    errors_bisection = [previous_total_diff]
    iteration_times = []
    for iteration in range(max_iter):
        iter_start_time = time.time()
        for i in range(n_params):
            l, u = 0, 1
            while (u - l) / 2 > tol:
                midpoint = (l + u) / 2
                params_left = params.copy()
                params_left[i] = midpoint - tol
                params_right = params.copy()
                params_right[i] = midpoint + tol

                if total_difference(params_left, x_points, y_points) < total_difference(params_right, x_points, y_points):
                    u = midpoint
                else:
                    l = midpoint
            params[i] = (l + u) / 2
        
        current_total_diff = total_difference(params, x_points, y_points)
        errors_bisection.append(current_total_diff)
        iteration_times.append(time.time() - iter_start_time)
        if abs(previous_total_diff - current_total_diff) < convergence_threshold:
            end_time = time.time()
            print(f"Convergence reached after {iteration + 1} iterations for bisection method.")
            print(f"Time taken to converge for Bisection is: {end_time - start_time:.4f} seconds")
            break
        previous_total_diff = current_total_diff

    return params, errors_bisection, iteration_times


# Optimized parameters using the BFGS method
def bfgs_optimization(params, x_points, y_points, convergence_threshold=1e-6):
    start_time = time.time()
    errors_bfgs = []
    iteration_times = []
    previous_error = total_difference(params, x_points, y_points)

    def callback(xk):
        nonlocal previous_error
        iter_start_time = time.time()
        error = total_difference(xk, x_points, y_points)
        errors_bfgs.append(error)
        iteration_times.append(time.time() - iter_start_time)
        if abs(previous_error - error) < convergence_threshold:
            return True
        previous_error = error

    result = minimize(total_difference, params, args=(x_points, y_points), method='BFGS', callback=callback, tol=convergence_threshold)
    end_time = time.time()
    print(f"Convergence reached after {result.nit} iterations for BFGS.")
    print(f"Time taken to converge for BFGS is: {end_time - start_time:.4f} seconds")
    return result.x, errors_bfgs, iteration_times

# Initial guess for the parameters
initial_params = generate_initial_params(p_indices, q_indices)

# Optimized parameters
optimized_params_bisection, errors_bisection, iteration_times_bisection = coordinate_bisection(initial_params.copy(), x_points, y_points)
optimized_params_bfgs, errors_bfgs, iteration_times_bfgs = bfgs_optimization(initial_params.copy(), x_points, y_points)

print("Initial parameters:")
print(initial_params)

print("Optimized parameters (Bisection):")
print(optimized_params_bisection)

print("Optimized parameters (BFGS):")
print(optimized_params_bfgs)

def optimized_rational_function(x, params):
    x1, x2 = x
    p = params[:len(p_indices)]
    q = params[len(p_indices):]
    
    num = sum([p[i] * (x1**idx[0]) * (x2**idx[1]) for i, idx in enumerate(p_indices)])
    den = sum([q[i] * (x1**idx[0]) * (x2**idx[1]) for i, idx in enumerate(q_indices)])
    
    rational_val1 = num / den
    
    return rational_val1


initial_y_points = np.array([optimized_rational_function((x1_points[i], x2_points[i]), initial_params) for i in range(len(x1_points))])

# Optimized y points
optimized_y_points_bisection = np.array([optimized_rational_function((x1_points[i], x2_points[i]), optimized_params_bisection) for i in range(len(x1_points))])
optimized_y_points_bfgs = np.array([optimized_rational_function((x1_points[i], x2_points[i]), optimized_params_bfgs) for i in range(len(x1_points))])

sse_bisection = np.sum((optimized_y_points_bisection - y_points)**2)
sse_bfgs = np.sum((optimized_y_points_bfgs - y_points)**2)

mse_bisection = sse_bisection / len(y_points)
mse_bfgs = sse_bfgs / len(y_points)

print(f"Sum of Squared Errors (SSE) for Bisection: {sse_bisection:.10e}")
print(f"Sum of Squared Errors (SSE) for BFGS: {sse_bfgs:.10e}")

print(f"Mean Squared Error (MSE) for Bisection: {mse_bisection:.10e}")
print(f"Mean Squared Error (MSE) for BFGS: {mse_bfgs:.10e}")


# Final differences
print("Initial vs Optimized differences (Bisection):")
optimized_differences_bisection = []
for i in range(len(x1_points)):
    initial_val1 = initial_y_points[i]
    optimized_val = optimized_y_points_bisection[i]
    actual_val = y_points[i]
    difference = (optimized_val - actual_val)**10
    optimized_differences_bisection.append(difference)
    print(f"Point ({x1_points[i]:.2f}, {x2_points[i]:.2f}): Initial Rational Value = {initial_val1:.10e}, Optimized Rational Value = {optimized_val:.10e}, Actual Value = {actual_val:.10e}, Difference = {difference:.10e}")

# Final differences
print("Initial vs Optimized differences (BFGS):")
optimized_differences_bfgs = []
for i in range(len(x1_points)):
    initial_val2 = initial_y_points[i]
    optimized_val = optimized_y_points_bfgs[i]
    actual_val = y_points[i]
    difference = (optimized_val - actual_val)**10
    optimized_differences_bfgs.append(difference)
    print(f"Point ({x1_points[i]:.2f}, {x2_points[i]:.2f}): Initial Rational Value = {initial_val2:.10e}, Optimized Rational Value = {optimized_val:.10e}, Actual Value = {actual_val:.10e}, Difference = {difference:.10e}")

# Plot for Bisection method
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
sc_bisection = ax.scatter(x1_points, x2_points, optimized_y_points_bisection, c=optimized_y_points_bisection, cmap='plasma', marker='o')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Optimized y points')
ax.set_title(r'Optimized y values using bisection method')
fig.colorbar(sc_bisection, ax=ax, format='%.2e')
fig.savefig('pgfs/optimized_y_points_bisection.pgf', bbox_inches='tight')
plt.show()

# Plot for BFGS method
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
sc_bfgs = ax.scatter(x1_points, x2_points, optimized_y_points_bfgs, c=optimized_y_points_bfgs, cmap='cool', marker='o')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Optimized y points')
ax.set_title(r'Optimized y values using BFGS method')
fig.colorbar(sc_bfgs, ax=ax, format='%.2e')
fig.savefig('pgfs/optimized_y_points_bfgs.pgf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 3.6))
plt.plot(errors_bisection, label='Bisection Method')
plt.plot(errors_bfgs, label='BFGS Method')
plt.xlabel('Number of iterations')
plt.ylabel('Final difference (r(x)-y)^10')
plt.yscale('log')
plt.title('Final difference at each iteration for both Bisection & BFGS Methods')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/final_difference_iterations.pgf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 3.6))
plt.plot(y_points, np.abs(optimized_y_points_bisection - y_points), 'o', label='Bisection method')
plt.xlabel('y value')
plt.ylabel('Error difference of y')
plt.title('Error difference of y using Bisection Method')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/error_difference_bisection.pgf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 3.6))
plt.plot(y_points, np.abs(optimized_y_points_bfgs - y_points), 's', label='BFGS Method')
plt.xlabel('y value')
plt.ylabel('Error difference of y')
plt.title('Error difference of y using BFGS Method')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/error_difference.pgf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 3.6))
plt.plot(np.abs(optimized_differences_bisection), 'o', label='Optimized difference of Bisection')
plt.plot(np.abs(optimized_differences_bisection), 'r', label='Bisection method')
plt.xlabel('Number of points')
plt.ylabel('Final difference (r(x)-y)^10')
plt.title('Optimized differences of bisection Method')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/optimized_differences_bisection.pgf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 3.6))
plt.plot(np.abs(optimized_differences_bfgs), 's', label='Optimized difference of BFGS')
plt.plot(np.abs(optimized_differences_bfgs), 'y', label='BFGS method')
plt.xlabel('Number of points')
plt.ylabel('Final difference (r(x)-y)^10')
plt.title('Optimized differences of BFGS Method')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/optimized_differences_bfgs.pgf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 3.6))
plt.plot(np.abs(optimized_differences_bisection), 'o', label='Bisection method')
plt.plot(np.abs(optimized_differences_bisection), 'r', label='Bisection method')
plt.plot(np.abs(optimized_differences_bfgs), 's', label='BFGS method')
plt.plot(np.abs(optimized_differences_bfgs), 'y', label='BFGS method')
plt.xlabel('Number of points')
plt.ylabel('Final difference (r(x)-y)^10')
plt.title('Optimized differences of Bisection & BFGS Method')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/optimized_differences.pgf', bbox_inches='tight')
plt.show()

bisection_iterations = range(len(iteration_times_bisection))
bisection_times = np.cumsum(iteration_times_bisection)

bfgs_iterations = range(len(iteration_times_bfgs))
bfgs_times = np.cumsum(iteration_times_bfgs)

plt.figure(figsize=(6, 3.6))
plt.plot(bisection_iterations, bisection_times, label='Bisection Method', marker='o')
plt.plot(bfgs_iterations, bfgs_times, label='BFGS Method', marker='s')
plt.xlabel('Number of iterations')
plt.ylabel('Time taken to converge (seconds)')
plt.title('Number of iterations vs Time taken to converge')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/iterations_vs_time.pgf', bbox_inches='tight')
plt.show()
