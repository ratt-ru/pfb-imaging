import numpy as np
import jax.numpy as jnp
import pyscilog
log = pyscilog.get_logger('FISTA')


def fista(value_and_grad, prox, x0, lam, gfunc=None,
          maxit=1000, tol=1e-4, report_freq=50,
          L0=1.0, eta=2.0, verbosity=1):
    """
    FISTA with backtracking line search for solving optimization problems of the form:
        min_x f(x) + g(x)
    where f is smooth and g has an easily computable proximal operator.
    
    Parameters:
    -----------
    value_and_grad : callable
        Function that returns the value and gradient of the smooth part f(x).
        Should have signature: (value, gradient) = value_and_grad(x)
    prox : callable
        Proximal operator for g(x) with step size t.
        Should have signature: prox_result = prox(x, t)
    x0 : ndarray
        Initial point.
    lam : float
        Regularization parameter for g(x).
    maxit : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criterion.
    L0 : float, optional
        Initial estimate of Lipschitz constant.
    eta : float, optional
        Multiplication factor for backtracking.
    verbose : bool, optional
        Whether to print progress information.
        
    Returns:
    --------
    x : ndarray
        Optimal solution found.
    
    """
    # Initialize variables
    x = np.copy(x0)
    y = np.copy(x0)
    t = 1.0
    L = L0
    obj_values = []
    
    # Initial function value and gradient
    f_y, grad_f_y = value_and_grad(y)
    
    for k in range(maxit):
        # Store previous point
        x_prev = np.copy(x)
        
        # Backtracking line search to find L
        while True:
            # Compute proximal gradient step
            grad_step = y - grad_f_y/L
            x_new = prox(grad_step, lam/L)
            
            # Compute function value at new point
            f_x_new, _ = value_and_grad(x_new)
            
            # Check sufficient decrease condition
            Q_L = f_y + np.dot(grad_f_y, x_new - y) + (L/2) * np.sum((x_new - y)**2)
            
            if f_x_new <= Q_L:
                break
            
            # Increase L and try again
            L = L * eta
            
            if verbosity and k % report_freq == 0:
                print(f"Iteration {k}, increasing L to {L}")
                
        # Update x with new point
        x = x_new
        
        # Compute new t
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        
        # Compute y for next iteration
        y = x + ((t - 1) / t_new) * (x - x_prev)
        
        # Update t
        t = t_new
        
        # Compute function value and gradient at new y
        f_y, grad_f_y = value_and_grad(y)
        
        # Compute objective value (f + g)
        if gfunc is not None:
            g_x = lam * gfunc(x)
            obj_value = f_x_new + g_x
            obj_values.append(obj_value)
            if verbosity and k % report_freq == 0:
                print(f"Iteration {k}, objective: {obj_value}, L: {L}")
        
        # Check convergence
        eps = np.linalg.norm(x - x_prev) / np.linalg.norm(x)
        if eps < tol:
            print(f"Converged after {k+1} iterations")
            break
    
    return x, obj_values
