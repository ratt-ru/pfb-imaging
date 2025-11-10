
# A general abstract base class for the Forward backward algorithm which works for arbitrary grad, prox
class FB():
    def __init__(self, grad, prox):
        pass

    # might be best to define these as abstract methods
    def grad(self):
        raise NotImplementedError

    def prox(self):
        raise NotImplementedError
    
    # this 
    def run(self, x):
        # define common forward and backward steps here
        pass