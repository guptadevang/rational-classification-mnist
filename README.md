# Rational-Classification

Please find the detailed report about the study [here](RClass_Report.pdf)


## Collaborators

### Devang Gupta
### Nayeemuddin Mohammed
### Shivakiran Nandala

## Rational Function
$$
r(x) = \frac{p(x)}{q(x)} = \frac{\sum_{|\alpha| \leq \alpha_{\text{num}}}^n p_{\alpha} x^{\alpha}}{\sum_{|\beta| \leq \beta_{\text{den}}}^n q_{\beta} x^{\beta}}
$$
                
Here,
$$
r: \mathbb{R}^2 \rightarrow \mathbb{R}
$$
$$
|\alpha| = \alpha_1 + \alpha_2
$$
$$
x^{\alpha} = x_1^{\alpha_1} x_2^{\alpha_2}
$$


## Optimization Objective

$$
\min_{p_k, q_k} \max_{k=1,\ldots,N} \left| y_i - \frac{p(x_i)}{q(x_i)} \right| = \left| y_i - \frac{\sum_{k=0}^{n} p_k x_{1}^{\alpha_1} x_{2}^{\alpha_2} }{\sum_{k=0}^{n} q_k x_{1}^{\beta_1} x_{2}^{\beta_2} } \right|
$$

## One Dimensional Results

### Bisection
![one_plot_bisection.png](images/one_plot_bisection.png)

### BFGS
![one_plot_bfgs.png](images/one_plot_bfgs.png)



## Multivariate Results


### Bisection


### BFGS


### Convergence Metrics


