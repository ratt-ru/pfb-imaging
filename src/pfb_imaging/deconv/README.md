# Deconvolution module

This module contains implementations of deconvolution algorithms corresponding to different regularisers.

## Deconvolution class structure

Each deconvolution class must perform any regulariser specific setup in the constructor.
It must also implement the following methods that will be run at every iteration:

* `first(residual)` - method to run first at every iteration (could be used to construct a mask for the forward step, for example.)
* `update = forward(residual)` - solve the forward step
* `set_grad(callable(x))` - set the gradient
* `model = backward(lam)` - solve the backward step
* `last(model, residual)` - method to run last at every iteration (could be used to update the prox, for example).
