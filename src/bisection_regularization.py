import sympy as sp
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.optimize import minimize

def f(x):
    x1, x2 = x
    return np.where((x1 + x2 < 1), 0, 1)

grid_size = 10 
x1_points, x2_points = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
x1_points = x1_points.flatten()
x2_points = x2_points.flatten()
x_points = np.vstack((x1_points, x2_points))

y_points = f(x_points)

def generate_indices(alpha, n):
    indices = []
    for comb in product(range(alpha + 1), repeat=n):
        if sum(comb) <= alpha:
            indices.append(comb)
    return indices

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

alpha_num = 10
alpha_den = 9
n = 2

rational_function, p_indices, q_indices, variables = generate_rational_function(alpha_num, alpha_den, n)
print("Generated rational function:")
print(rational_function)

def difference_function(x, params):
    x1, x2 = x
    p = params[:len(p_indices)]
    q = params[len(p_indices):]
    
    num = sum([p[i] * (x1**idx[0]) * (x2**idx[1]) for i, idx in enumerate(p_indices)])
    den = sum([q[i] * (x1**idx[0]) * (x2**idx[1]) for i, idx in enumerate(q_indices)])
    
    rational_val = num / den
    actual_val = f((x1, x2))
    difference = (rational_val - actual_val)
    
    return rational_val, difference

def total_difference(params, x_points, y_points, regularization_lambda=0.1, include_regularization=True):
    total_diff = 0
    for i in range(x_points.shape[1]):
        _, diff = difference_function((x_points[0, i], x_points[1, i]), params)
        total_diff += diff**10

    if include_regularization:
        regularization_term = regularization_lambda * np.sum(params**2)
        total_diff += regularization_term
    
    return total_diff

def generate_initial_params(p_indices, q_indices):
    initial_params = [1] * len(p_indices) + [0.1] * len(q_indices)
    return np.array(initial_params)

def coordinate_bisection(params, x_points, y_points, tol=1e-6, max_iter=1000, convergence_threshold=1e-5, regularization_lambda=0.1):
    start_time = time.time()
    n_params = len(params)
    previous_total_diff = total_difference(params, x_points, y_points, regularization_lambda)
    errors_bisection = [previous_total_diff]
    for iteration in range(max_iter):
        for i in range(n_params):
            l, u = 0, 1
            while (u - l) / 2 > tol:
                midpoint = (l + u) / 2
                params_left = params.copy()
                params_left[i] = midpoint - tol
                params_right = params.copy()
                params_right[i] = midpoint + tol

                if total_difference(params_left, x_points, y_points, regularization_lambda) < total_difference(params_right, x_points, y_points, regularization_lambda):
                    u = midpoint
                else:
                    l = midpoint
            params[i] = (l + u) / 2
        
        current_total_diff = total_difference(params, x_points, y_points, regularization_lambda)
        errors_bisection.append(current_total_diff)
        if abs(previous_total_diff - current_total_diff) < convergence_threshold:
            end_time = time.time()
            print(f"Convergence reached after {iteration + 1} iterations.")
            print(f"Time taken to converge (Bisection): {end_time - start_time:.4f} seconds")
            break
        previous_total_diff = current_total_diff

    return params, errors_bisection

def bfgs_optimization(params, x_points, y_points, regularization_lambda=0.1, convergence_threshold=1e-6):
    start_time = time.time()
    errors_bfgs = []
    previous_error = total_difference(params, x_points, y_points, regularization_lambda)

    def callback(xk):
        nonlocal previous_error
        error = total_difference(xk, x_points, y_points, regularization_lambda)
        errors_bfgs.append(error)
        if abs(previous_error - error) < convergence_threshold:
            return True
        previous_error = error

    result = minimize(total_difference, params, args=(x_points, y_points, regularization_lambda), method='BFGS', callback=callback, tol=convergence_threshold)
    end_time = time.time()
    print(f"Convergence reached after {result.nit} iterations (BFGS).")
    print(f"Time taken to converge (BFGS): {end_time - start_time:.4f} seconds")
    return result.x, errors_bfgs

initial_params = generate_initial_params(p_indices, q_indices)

optimized_params_bisection, errors_bisection = coordinate_bisection(initial_params.copy(), x_points, y_points)

optimized_params_bfgs, errors_bfgs = bfgs_optimization(initial_params.copy(), x_points, y_points)

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

optimized_y_points_bisection = np.array([optimized_rational_function((x1_points[i], x2_points[i]), optimized_params_bisection) for i in range(len(x1_points))])
optimized_y_points_bfgs = np.array([optimized_rational_function((x1_points[i], x2_points[i]), optimized_params_bfgs) for i in range(len(x1_points))])

sse_bisection = np.sum((optimized_y_points_bisection - y_points)**2)
sse_bfgs = np.sum((optimized_y_points_bfgs - y_points)**2)

mse_bisection = sse_bisection / len(y_points)
mse_bfgs = sse_bfgs / len(y_points)

sse_bisection_no_reg = total_difference(optimized_params_bisection, x_points, y_points, include_regularization=False)
sse_bfgs_no_reg = total_difference(optimized_params_bfgs, x_points, y_points, include_regularization=False)

print(f"Sum of Squared Errors (SSE) for Bisection: {sse_bisection:.10e}")
print(f"Sum of Squared Errors (SSE) for BFGS: {sse_bfgs:.10e}")

print(f"Mean Squared Error (MSE) for Bisection: {mse_bisection:.10e}")
print(f"Mean Squared Error (MSE) for BFGS: {mse_bfgs:.10e}")

print(f"Sum of Squared Errors (SSE) without regularization for Bisection: {sse_bisection_no_reg:.10e}")
print(f"Sum of Squared Errors (SSE) without regularization for BFGS: {sse_bfgs_no_reg:.10e}")

print("Individual differences (Bisection):")
for i in range(len(x1_points)):
    rational_val, difference = difference_function((x1_points[i], x2_points[i]), optimized_params_bisection)
    print(f"Point ({x1_points[i]:.2f}, {x2_points[i]:.2f}): Rational Value = {rational_val:.10e}, Actual Value = {f((x1_points[i], x2_points[i])):.10e}, {x2_points[i]:.2f}): Predicted Y value = {rational_val:.10e}, Difference = {difference:.10e}")

print("Individual differences (BFGS):")
for i in range(len(x1_points)):
    rational_val, difference = difference_function((x1_points[i], x2_points[i]), optimized_params_bfgs)
    print(f"Point ({x1_points[i]:.2f}, {x2_points[i]:.2f}): Rational Value = {rational_val:.10e}, Actual Value = {f((x1_points[i], x2_points[i])):.10e}, {x2_points[i]:.2f}): Predicted Y value = {rational_val:.10e}, Difference = {difference:.10e}")

print("Initial vs Optimized differences (Bisection):")
initial_differences_bisection = []
optimized_differences_bisection = []
for i in range(len(x1_points)):
    initial_val = initial_y_points[i]
    optimized_val = optimized_y_points_bisection[i]
    actual_val = y_points[i]
    initial_difference = initial_val - actual_val
    optimized_difference = optimized_val - actual_val
    initial_differences_bisection.append(initial_difference)
    optimized_differences_bisection.append(optimized_difference)
    print(f"Point ({x1_points[i]:.2f}, {x2_points[i]:.2f}): Initial Rational Value = {initial_val:.10e}, Optimized Rational Value = {optimized_val:.10e}, Actual Value = {actual_val:.10e}")
    print(f"Point ({x1_points[i]:.2f}, {x2_points[i]:.2f}):Initial Difference = {initial_difference:.10e}, Optimized Difference = {optimized_difference:.10e}")
    
    
print("Initial vs Optimized differences (BFGS):")
initial_differences_bfgs = []
optimized_differences_bfgs = []
for i in range(len(x1_points)):
    initial_val = initial_y_points[i]
    optimized_val = optimized_y_points_bfgs[i]
    actual_val = y_points[i]
    initial_difference = initial_val - actual_val
    optimized_difference = optimized_val - actual_val
    initial_differences_bfgs.append(initial_difference)
    optimized_differences_bfgs.append(optimized_difference)
    print(f"Point ({x1_points[i]:.2f}, {x2_points[i]:.2f}): Initial Rational Value = {initial_val:.10e}, Optimized Rational Value = {optimized_val:.10e}, Actual Value = {actual_val:.10e}")
    print(f"Point ({x1_points[i]:.2f}, {x2_points[i]:.2f}): Initial Difference = {initial_difference:.10e}, Optimized Difference = {optimized_difference:.10e}")


fig = plt.figure(figsize=(14, 7))

ax = fig.add_subplot(121, projection='3d')
sc_bisection = ax.scatter(x1_points, x2_points, optimized_y_points_bisection, c=optimized_y_points_bisection, cmap='plasma', marker='o')
#ax.plot(x1_points, x2_points, optimized_y_points_bisection, color='red')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Optimized Y value')
ax.set_title('Optimized Y value of F(x) (Bisection)')
fig.colorbar(sc_bisection, ax=ax, format='%.2e')


ax = fig.add_subplot(122, projection='3d')
sc_bfgs = ax.scatter(x1_points, x2_points, optimized_y_points_bfgs, c=optimized_y_points_bfgs, cmap='cool', marker='o')
#ax.plot(x1_points, x2_points, optimized_y_points_bfgs, color='blue')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Optimized Y value')
ax.set_title('Optimized Y value of F(x) (BFGS)')
fig.colorbar(sc_bfgs, ax=ax, format='%.2e')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(errors_bisection, label='Bisection Method')
plt.plot(errors_bfgs, label='BFGS Method')
plt.xlabel('Iterations')
plt.ylabel('Sum of all Difference')
plt.yscale('log')
plt.title('Error Comparison of Bisection vs BFGS for Sum of all Difference [(rational_val - actual_val)**10] vs Iteration')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.abs(initial_differences_bisection), 'o', label='Initial each Differences (Bisection)')
plt.plot(np.abs(optimized_differences_bisection), 'x', label='Optimized each Differences (Bisection)')
plt.xlabel('Number of Points')
plt.ylabel('Difference')
plt.title('Comparison of Initial and Optimized Differences (Bisection)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(np.abs(initial_differences_bfgs), 's', label='Initial each Differences (BFGS)')
plt.plot(np.abs(optimized_differences_bfgs), 'd', label='Optimized each Differences (BFGS)')
plt.xlabel('Number of Points')
plt.ylabel('Difference')
plt.title('Comparison of Initial and Optimized Differences (BFGS)')
plt.legend()
plt.grid(True)
plt.show()