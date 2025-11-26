import numpy as np

from .forward_backward import ForwardBackward


class ISTA(ForwardBackward):
    """
        Implement a Forward Backward class that solves optimization problem of the form

        $$
            min_x f(x) + g(x)
        $$

        The iterates are of the form
        $$
        x^{k+1} = prox_{\gamma g} ( x^k - \gamma \nabla f(x^k))
        $$



    """
    def __init__(self, lmbda, max_iter, precond, step_size=None,tol=1e-12, log_iter=10, return_metrics=False):
        
        super().__init__(max_iter=max_iter,step_size=step_size, tol=tol, log_iter=log_iter, return_metrics=return_metrics)
   
        self.lmbda = lmbda # Threshold on L1 norm
        self.precond = precond

            
    def __str__(self):
        return "ISTA"

    
    def _gradf(self, x, y):
        """
            Implement the gradient of the smooth term of the objective function
            
            $$
                \nabla_x f(x,y)
            $$
        """
        return -self.precond.dot(y - x)/self.lmbda

    def _proxg(self, x, gamma):

        """ 
            Implements prox_{\gamma g} = \argmin_u \frac{1}{2\gamma}\| x - u \|_2^2 + g(x)

            When g(x) = \|x\|_1, prox g is a soft thresholding with threshold gamma.
        """
        return np.max(x - np.abs(gamma), 0)

    def _obj_fun(self, x, y):
        """
            Implement the objective function explicitely 
            $$
                \| x - y \|_U + \|x \|_1
            $$
        """
        val = (1/self.lmbda) * (x-y).T.conjugate() @ self.precond.dot(x-y) + np.sum(np.abs(x))
        return val.squeeze()
    
    def _rel_var(self, x, x_prev):
        return np.sum(x - x_prev)**2 / np.sum(x_prev**2)


