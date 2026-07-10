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
class LinearOperator(Protocol):
    """Hermitian image-space operator (Hessian family). Allocating style.

    Methods:

    - dot: apply the operator; returns a new array
    - hdot: apply the adjoint (same as dot for Hermitian operators)
    """

    def dot(self, x): ...

    def hdot(self, x): ...


@runtime_checkable
class PsiOperator(Protocol):
    """Analysis/synthesis operator pair. In-place style.

    Shape attributes are part of the contract; coefficient buffers have
    shape ``(nband, nbasis, nymax, nxmax)`` and images ``(nband, nx, ny)``.

    Methods:

    - dot: analysis, image to coefficients, fills ``alphao`` in-place
    - hdot: synthesis, coefficients to image, fills ``xo`` in-place
    """

    nband: int
    nbasis: int
    nymax: int
    nxmax: int
    nx: int
    ny: int

    def dot(self, x, alphao): ...

    def hdot(self, alpha, xo): ...


def _protocol_members(protocol) -> set:
    """Public members a runtime_checkable Protocol requires (methods + annotated attrs)."""
    members = {name for name in getattr(protocol, "__annotations__", {}) if not name.startswith("_")}
    for name, value in vars(protocol).items():
        if not name.startswith("_") and callable(value):
            members.add(name)
    return members


def require_protocol(obj, protocol, arg_name: str) -> None:
    """Raise a clear TypeError when ``obj`` does not satisfy a runtime_checkable Protocol.

    Args:
        obj: Object to validate.
        protocol: The ``@runtime_checkable`` Protocol class to validate against.
        arg_name: Argument name to use in the error message.

    Raises:
        TypeError: Naming the Protocol and the members ``obj`` is missing.
    """
    if isinstance(obj, protocol):
        return
    missing = sorted(m for m in _protocol_members(protocol) if not hasattr(obj, m))
    detail = f"is missing: {', '.join(missing)}" if missing else "does not conform"
    raise TypeError(f"{arg_name} must satisfy the {protocol.__name__} Protocol; {type(obj).__name__} {detail}")
