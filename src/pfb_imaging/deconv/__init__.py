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

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DeconvSolver(Protocol):
    """Protocol for deconvolution solvers used within the outer PFB loop.

    Each solver is responsible for all regulariser-specific setup in
    ``__init__`` and implements the forward/backward split of the
    preconditioned forward-backward (PFB) algorithm.

    The outer loop calls these methods in order each major iteration::

        solver.first(residual)
        update = solver.forward(residual)
        lam = rmsfactor * rms           # computed externally
        model = solver.backward(lam)
        solver.last()
        residual = compute_residual(model)  # gridder, always external
    """

    def first(self, residual: np.ndarray) -> None:
        """Per-iteration preprocessing (e.g. beam application)."""
        ...

    def forward(self, residual: np.ndarray) -> np.ndarray:
        """Solve the forward (preconditioned gradient) step.

        Returns:
            update: preconditioned gradient image, shape ``(nband, nx, ny)``.
        """
        ...

    def backward(self, lam: float) -> np.ndarray:
        """Solve the backward (proximal) step.

        Args:
            lam: Regularisation strength for this outer iteration.

        Returns:
            model: updated model image, shape ``(nband, nx, ny)``.
        """
        ...

    def last(self) -> None:
        """Per-iteration post-processing (e.g. L1 weight updates)."""
        ...
