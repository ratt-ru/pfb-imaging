from copy import deepcopy

# A general abstract base class for the Forward backward algorithm which works for arbitrary grad, prox


class ForwardBackward():
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

    def __str__(self):
        return "Generic Forward Backward"

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def __init__(self, max_iter, step_size=None,tol=1e-12, log_iter=10, return_metrics=False):
        
        self.step_size = step_size if step_size is not None else self._get_step_size()
        self.max_iter = max_iter
        self.tol = tol
        self.return_metrics = return_metrics
        self.log_iter = log_iter
    
    def _get_step_size():
        pass
    
    def _gradf(self, x, y):
        """
            Implement the gradient of the smooth term of the objective function
            
            $$
                \nabla_x f(x,y)
            $$
        """
        pass

    def _proxg(self, x, gamma):

        """ 
            Implements prox_{\gamma g} = \argmin_u \frac{1}{2\gamma}\| x - u \|_2^2 + g(x)
        
        """

        pass

    def _obj_fun(self, x, y):
        """
            Implement the objective function explicitely 
            $$
                 f(x,y) + g(x)
            $$
        """

        pass

    def _rel_var(self, x, x_prev):
        return sum(x - x_prev)**2 / sum(x_prev**2)
    
    def _convergence_criteria(self, obj_fun, rel_var):
        return rel_var < self.tol


    def _initial_solution(self, y):
        return y


    def _schedule(self, x, x_prev, y, iteration):
        """
            No scheduling by default.
        """
        # self.step_size = self.step_size / 2 ...
        # ...
        pass

    def _log(self, string):
        print(string) # by default return string to stdout.

    def _get_metrics(self, x, x_prev, y):
        """
        
        By default compute the relative variation (using python standard sum function) and the objective function.
        """

        rel_var = self.rel_var(x,x_prev)
        obj_fun = self.obj_fun(x,y)
        return {'rel_var' : rel_var, 
                'obj_fun' : obj_fun }

    def forward(self, y):

        self._log(f"Running an instance of the {self.__str__} algorithm.") ## Add more info ?
       
        self._log(f"[{self.__str__}] Initializing products.")
        x = self._initial_solution(y) 

        metrics = {} # Initialize a dictionary object to store metrics (run time, objective function, relative variation, rms, ...) 
        obj_fun = self.obj_fun(x, y)
        rel_var = float('inf') 

        it = 0
        while (it < self.max_iter and self._convergence_criteria(rel_var, obj_fun)):

            if it%self.log_iter:
                self._log(f"[{self.__str__}] Running iteration {it}; max_iter = {self.max_iter}.")

            ## Deepcopy the previous estimate to compute convergence criteria
            x_prev = deepcopy(x)

            ## The FB iteration
            gradf = self._gradf(x, y) 
            x = self._proxg(x - self.step_size * gradf, gamma=self.step_size)


            if it%self.log_iter:
                ## Compute the current metrics and append them to the global metrics dictionnary

                current_metrics = self._get_metrics(x, x_prev, y)
                            
                self._log(f"[{self.__str__}] Iteration : ")
                for key, item in current_metrics.items():
                    self._log(f"[{self.__str__}] \t {key} : {item} ")

                assert ("rel_var" in current_metrics.keys()) and ("obj_fun" in current_metrics.keys())
                rel_var, obj_fun = current_metrics["rel_var"], current_metrics["obj_fun"] # at least the relative variation and the objective function should be computed.

                for key, item in current_metrics.items():
                    if not (key in metrics.keys()):
                        metrics[key] = []
                    metrics[key].append(item)

            ## Are they hyperparameters to schedule given iteration, data, current and previous estimate ?
            self._schedule(x, x_prev, y, iteration=it)
            it += 1 
        
        if it < self.max_iter:
            self._log(f"[{self.__str__}] Optimisation complete. Converged after {it} iterations.")
        else:
            self._log(f"[{self.__str__}] Optimisation complete. Stoped after {it} iterations without achieving convergence.")

        # metrics['runtime'] = 
        # if self.save_metrics:
            ## save metrics in .json

        if self.return_metrics:
            return x, metrics
        else:
            return x


