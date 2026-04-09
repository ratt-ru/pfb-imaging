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

# defines operator protocols

from typing import Protocol, runtime_checkable


@runtime_checkable
class Preconditioner(Protocol):
    """
    The preconditioner needs to be Hermitian and invertible.
    Methods:

    - dot: applies the preconditioner to a vector
    - hdot: applies the adjoint of dot (same as dot for Hermitian operators)
    - idot: applies the inverse of the preconditioner to a vector
    """

    def dot(self, x): ...

    def hdot(self, x): ...

    def idot(self, x): ...


@runtime_checkable
class PsiOperatorProtocol(Protocol):
    """
    The signal decomposition operator only needs to be able to apply the operator and its adjoint.
    Methods:

    - dot: applies the operator to a vector
    - hdot: applies the adjoint of the operator to a vector
    """

    def dot(self, x, alphao): ...

    def hdot(self, alpha, xo): ...


@runtime_checkable
class ProxOperatorProtocol(Protocol):
    """
    The proximal operator needs to be able to apply the operator and its adjoint.
    Methods:

    - prox: applies the proximal operator to a vector
    - hprox: applies the adjoint of the proximal operator to a vector
    """

    def prox(self, x, rho): ...

    def hprox(self, x, rho): ...
