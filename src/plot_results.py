import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure Matplotlib to use xelatex
mpl.rcParams.update({
    "pgf.texsystem": "xelatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
})

# Inputs
alpha_values = ['2/1', '2/2', '3/1', '3/2', '3/3', '4/1', '4/2', '4/3', '4/4', '5/1', '5/2', '5/3', '5/4']
bisection_iterations = [39, 48, 71, 14, 128, 40, 72, 184, 58, 51, 350, 563, 13]
bisection_time = [29.5762, 113.1742, 187.2826, 26.6981, 663.2610, 175.9013, 410.6712, 1406.8333, 527.9268, 353.7957, 4088.2029, 1540.5217, 46.1245]

bfgs_iterations = [172, 119, 314, 285, 146, 589, 337, 297, 1, 366, 790, 595, 249]
bfgs_time = [2.5385, 4.1647, 12.3491, 7.3302, 15.4772, 37.8888, 26.3675, 30.0009, 0.3216, 34.8862, 118.0610, 22.2017 ,13.2796]

bisection_sse = [8.9250686172, 8.7458516086, 9.3879330561, 9.0424704567, 7.7389526053, 9.7009616780, 8.9249664770, 7.7483256331, 7.3830613447, 9.5879750274, 1.0235778585, 8.4626420489e+00, 8.2013175384e+00]
bfgs_sse = [7.9299091273, 4.0428799978, 5.6687837895, 2.6434234615, 2.9194365015, 5.9067530393, 2.3775728805, 1.9664390065, 3.6775540714, 5.7713564435, 5.5990580743, 1.8787839470e+00, 3.9994456344e+00]

bisection_mse = [8.9250686172e-02, 8.7458516086e-02, 9.3879330561e-02, 9.0424704567e-02, 7.7389526053e-02, 9.7009616780e-02, 8.9249664770e-02, 7.7483256331e-02, 7.3830613447e-02, 9.5879750274e-02, 1.0235778585e-01, 8.4626420489e-02, 8.2013175384e-02]
bfgs_mse = [7.9299091273e-02, 4.0428799978e-02, 5.6687837895e-02, 2.6434234615e-02, 2.9194365015e-02, 5.9067530393e-02, 2.3775728805e-02, 1.9664390065e-02, 3.6775540714e-01, 5.7713564435e-02, 5.5990580743e-02, 1.8787839470e-02, 3.9994456344e-02]

# Plots

plt.figure(figsize=(6, 3.6))
plt.plot(alpha_values, bisection_time, marker='o', linestyle='-', color='r', label='Bisection Time (seconds)')
plt.plot(alpha_values, bfgs_time, marker='x', linestyle='--', color='m', label='BFGS Time (seconds)')
plt.xlabel('Alpha values')
plt.ylabel('Time (seconds)')
plt.title('Alpha values vs Convergence (Time) for Bisection and BFGS methods')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/alpha_convergence_time.pgf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 3.6))
plt.plot(alpha_values, bisection_iterations, marker='s', linestyle='-', color='y', label='Bisection Iterations')
plt.plot(alpha_values, bfgs_iterations, marker='v', linestyle='--', color='b', label='BFGS Iterations')
plt.xlabel('Alpha Values')
plt.ylabel('Iterations')
plt.title('Alpha Values vs. No. of Iterations for Bisection and BFGS Methods')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/alpha_num_iterations.pgf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 3.6))
plt.plot(alpha_values, bisection_mse, marker='o', linestyle='-', color='k', label='Bisection mean square error')
plt.plot(alpha_values, bfgs_mse, marker='x', linestyle='--', color='#FFA500', label='BFGS mean square error')
plt.xlabel('Alpha Values')
plt.ylabel('Mean square error')
plt.title('Alpha Values vs. MSE for Bisection and BFGS Methods')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/alpha_mse.pgf', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6, 3.6))
plt.plot(alpha_values, bisection_sse, marker='s', linestyle='-', color='g', label='Bisection sum of squared error')
plt.plot(alpha_values, bfgs_sse, marker='v', linestyle='--', color='c', label='BFGS sum of squared error')
plt.xlabel('Alpha Values')
plt.ylabel('Sum of squared error')
plt.title('Alpha Values vs. SSE for Bisection and BFGS Methods')
plt.legend()
plt.grid(True)
plt.savefig('pgfs/alpha_sse.pgf', bbox_inches='tight')
plt.show()