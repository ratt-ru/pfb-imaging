
# Future to unify sara/airi/resolve etc.
class BasePFBAlgorithm():
    def __init__(self, init_paranms):
        pass

    def gradient(self, x):
        pass

    def forward(self, x):
        pass

    def backward(self, x):
        pass

    def update(self, x):
        # this could be used for hyper-parameter selection
        pass

    def save_model(self, x):
        pass