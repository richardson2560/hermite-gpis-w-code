#hgw/core/kernels.py

import numpy as np

def wendland_c2_profile(u):
    """
    Compute Wendland's C2 kernel profile and its first and second derivatives.
    """
    u = np.asarray(u, dtype=np.float64)
    phi = np.zeros_like(u)
    dphi = np.zeros_like(u)
    d2phi = np.zeros_like(u)

    mask = u < 1.0
    v = u[mask]
    one_minus = 1.0 - v

    phi[mask] = one_minus ** 4 * (4.0 * v + 1.0)
    dphi[mask] = -20.0 * v * one_minus ** 3
    d2phi[mask] = 20.0 * one_minus ** 2 * (4.0 * v - 1.0)

    return phi, dphi, d2phi

def wendland_c2_value(r, h, sigma_f2):
    """
    Compute the Wendland C2 kernel value for a given distance r, bandwidth h, and scaling factor sigma_f2.
    """
    u = np.asarray(r, dtype=np.float64) / h
    phi, _, _ = wendland_c2_profile(u)
    return sigma_f2 * phi

def wendland_c2_gradient(d, r, h, sigma_f2):
    """
    Compute the gradient of the Wendland C2 kernel for a given distance r, bandwidth h, and scaling factor sigma_f2.
    """
    if r == 0.0:
        return np.zeros_like(d)
    if r >= h:
        return np.zeros_like(d)

    return (-20.0 * sigma_f2 / h**2) * (1.0 - r / h)**3 * d

def wendland_c2_hessian(d, r, h, sigma_f2):
    """
    Compute the Hessian of the Wendland C2 kernel for a given distance r, bandwidth h, and scaling factor sigma_f2.
    """
    if r == 0.0:
        return np.eye(3, dtype=np.float64) * (-20.0 * sigma_f2 / h**2)
    if r >= h:
        return np.zeros((3, 3), dtype=np.float64)

    _, dphi, d2phi = wendland_c2_profile(np.array([r / h]))
    k_prime = sigma_f2 * dphi[0] / h
    k_second = sigma_f2 * d2phi[0] / (h**2)
    d_outer = np.outer(d, d) / r**2
    H = k_second * d_outer + k_prime * (np.eye(3, dtype=np.float64) - d_outer) / r

    return H

def wendland_c2_hessian_vector_product(d, r, n, h, sigma_f2):
    """Return Hessian(k) @ n without a matrix or singular radial factors."""
    n = np.asarray(n, dtype=np.float64)
    if r >= h:
        return np.zeros_like(n)
    scale = 20.0 * sigma_f2 / h**2
    if r == 0.0:
        return -scale * n
    u = r / h
    direction = np.asarray(d, dtype=np.float64) / r
    return scale * (1.0 - u)**2 * (
        -(1.0 - u) * n + 3.0 * u * direction * np.dot(direction, n)
    )
