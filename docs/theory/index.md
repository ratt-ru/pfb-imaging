# Mathematical Background

This section provides the mathematical foundation for the algorithms implemented in pfb-imaging.

## Radio Interferometry Forward Model

The radio interferometric measurement equation relates the observed visibilities to the sky brightness distribution:

$$
V_{ij}(u,v,w) = \int\int I(\ell, m) \cdot A_i(\ell, m) \cdot A_j^*(\ell, m) \cdot e^{2\pi i (u\ell + vm + w(\sqrt{1-\ell^2-m^2}-1))} \, d\ell \, dm
$$

Where:
- $V_{ij}$ is the visibility measured by antennas $i$ and $j$
- $I(\ell, m)$ is the sky brightness distribution
- $A_i(\ell, m)$ is the primary beam pattern of antenna $i$
- $(u,v,w)$ are the baseline coordinates
- $(\ell, m)$ are direction cosines

## Discretization

In the discrete case, we can write the measurement equation as:

$$
\vec{v} = A \vec{x} + \vec{n}
$$

Where:
- $\vec{v}$ are the observed visibilities
- $\vec{x}$ is the discretized sky image
- $A$ is the measurement operator
- $\vec{n}$ is the noise

## Optimization Problem

The image reconstruction problem is formulated as:

$$
\hat{\vec{x}} = \arg\min_{\vec{x}} \frac{1}{2} \|A\vec{x} - \vec{v}\|^2_2 + \lambda R(\vec{x})
$$

Where:
- $\frac{1}{2} \|A\vec{x} - \vec{v}\|^2_2$ is the data fidelity term
- $R(\vec{x})$ is a regularization term
- $\lambda$ controls the regularization strength

## Common Regularization Terms

### L1 Sparsity
$$
R(\vec{x}) = \|\vec{x}\|_1 = \sum_i |x_i|
$$

### Wavelet Sparsity
$$
R(\vec{x}) = \|\Psi\vec{x}\|_1
$$
where $\Psi$ is a wavelet transform.

### Total Variation
$$
R(\vec{x}) = \|\nabla \vec{x}\|_1
$$

## Algorithm Templates

### Template: Iterative Algorithm

```python
def iterative_algorithm(A, b, x0, maxiter=100, tol=1e-6):
    """
    Template for iterative reconstruction algorithms.
    
    Parameters
    ----------
    A : LinearOperator
        Measurement operator
    b : np.ndarray
        Observed data
    x0 : np.ndarray
        Initial estimate
    maxiter : int
        Maximum iterations
    tol : float
        Convergence tolerance
        
    Returns
    -------
    x : np.ndarray
        Reconstructed image
    """
    x = x0.copy()
    
    for k in range(maxiter):
        # Compute residual
        r = A(x) - b
        
        # Update step (algorithm-specific)
        x = update_step(x, r, A)
        
        # Check convergence
        if np.linalg.norm(r) < tol:
            break
            
    return x
```

### Template: Proximal Algorithm

```python
def proximal_algorithm(A, b, prox_R, gamma, maxiter=100):
    """
    Template for proximal algorithms.
    
    Mathematical formulation:
    
    x^{k+1} = prox_{γR}(x^k - γ∇f(x^k))
    
    where f(x) = (1/2)||Ax - b||²
    
    Parameters
    ----------
    A : LinearOperator
        Measurement operator
    b : np.ndarray
        Observed data
    prox_R : callable
        Proximal operator of regularization R
    gamma : float
        Step size
    maxiter : int
        Maximum iterations
        
    Returns
    -------
    x : np.ndarray
        Solution
    """
    x = np.zeros_like(b)
    
    for k in range(maxiter):
        # Gradient step
        grad = A.adjoint(A(x) - b)
        y = x - gamma * grad
        
        # Proximal step
        x = prox_R(y, gamma)
        
    return x
```

### Template: Preconditioned Algorithm

```python
def preconditioned_algorithm(A, b, P, maxiter=100):
    """
    Template for preconditioned algorithms.
    
    The preconditioned system is:
    P⁻¹Ax = P⁻¹b
    
    Parameters
    ----------
    A : LinearOperator
        Measurement operator
    b : np.ndarray
        Observed data
    P : LinearOperator
        Preconditioner
    maxiter : int
        Maximum iterations
        
    Returns
    -------
    x : np.ndarray
        Solution
    """
    def preconditioned_operator(x):
        return P.solve(A(x))
    
    def preconditioned_rhs():
        return P.solve(b)
    
    # Solve preconditioned system
    x = solve_linear_system(preconditioned_operator, preconditioned_rhs())
    
    return x
```

## Performance Considerations

### Computational Complexity

| Algorithm | Per Iteration | Total |
|-----------|---------------|--------|
| Gradient Descent | $O(N \log N)$ | $O(K \cdot N \log N)$ |
| Conjugate Gradient | $O(N \log N)$ | $O(\sqrt{\kappa} \cdot N \log N)$ |
| FISTA | $O(N \log N)$ | $O(\sqrt{L/\lambda} \cdot N \log N)$ |

Where:
- $N$ is the number of image pixels
- $K$ is the number of iterations
- $\kappa$ is the condition number
- $L$ is the Lipschitz constant

### Memory Requirements

| Component | Memory Usage |
|-----------|---------------|
| Image | $O(N)$ |
| Visibilities | $O(M)$ |
| Gridding Kernel | $O(N)$ |
| PSF | $O(N)$ |

Where $M$ is the number of visibilities.

## Mathematical Properties

### Convexity
The optimization problem is convex if:
1. The measurement operator $A$ is linear
2. The regularization term $R(\vec{x})$ is convex

### Convergence Guarantees
For convex problems with Lipschitz gradient:
- Gradient descent: $O(1/k)$ convergence
- Accelerated methods: $O(1/k^2)$ convergence
- Proximal methods: $O(1/k)$ convergence

### Optimality Conditions
The solution satisfies the first-order optimality condition:
$$
A^T(A\hat{\vec{x}} - \vec{v}) + \lambda \partial R(\hat{\vec{x}}) \ni 0
$$

Where $\partial R(\hat{\vec{x}})$ is the subdifferential of $R$ at $\hat{\vec{x}}$.

## Implementation Guidelines

### Numerical Stability
- Use appropriate data types (float64 for coordinates)
- Implement proper scaling for different units
- Handle edge cases in algorithms

### Performance Optimization
- Use vectorized operations where possible
- Leverage FFT for convolution operations
- Implement chunked processing for large datasets

### Testing Mathematical Properties
```python
def test_adjoint_property(A, x, y):
    """Test that <Ax, y> = <x, A*y>."""
    lhs = np.vdot(A(x), y)
    rhs = np.vdot(x, A.adjoint(y))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)

def test_operator_norm(A, x):
    """Test operator norm properties."""
    norm_Ax = np.linalg.norm(A(x))
    norm_x = np.linalg.norm(x)
    operator_norm = norm_Ax / norm_x
    return operator_norm
```

## See Also

- [Forward-Backward Algorithm](forward-backward.md) - Detailed algorithm description
- [Preconditioning](preconditioning.md) - Acceleration techniques
- [Sparsity Regularization](sparsity.md) - Sparsity-based methods
- [Optimization](optimization.md) - General optimization theory