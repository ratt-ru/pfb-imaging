"""
pfb-imaging

author - Landman Bester
email  - lbester@ska.ac.za
date   - 31/03/2020

MIT License

Copyright (c) 2020 Rhodes University Centre for Radio Astronomy Techniques & Technologies (RATT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

# defines optimiser protocols

from typing import Protocol, runtime_checkable


@runtime_checkable
class ForwardSolver(Protocol):
    """Solves the forward (preconditioned gradient) step.

    Methods:

    - solve: return ``update ≈ hess^{-1} residual``; ``hess`` satisfies
      the ``LinearOperator`` Protocol and is passed per call
    """

    def solve(self, hess, residual, x0=None): ...


@runtime_checkable
class BackwardSolver(Protocol):
    """Solves the backward (proximal) step.

    Lifecycle: constructor takes algorithm options only; ``setup`` binds
    the regulariser and step sizes once; ``set_grad`` is called each major
    cycle; ``solve`` iterates.  Auxiliary state (e.g. a dual variable) is
    internal and warm-started across calls; ``reset`` drops it.

    Methods:

    - setup: bind regulariser and hessnorm, size buffers
    - set_grad: set the gradient of the smooth data-fidelity term
    - solve: run the solve loop, return the updated x
    - reset: drop warm-start state
    """

    def setup(self, prox, hessnorm): ...

    def set_grad(self, grad): ...

    def solve(self, x, lam): ...

    def reset(self): ...
