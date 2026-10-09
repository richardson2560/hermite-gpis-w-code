"""Stationary radial kernels and linear observation operators.

d = x - p; Cov(f(x), D_n f(p)) = -grad_x(k).n.
Full-gradient observations have four channels per point.
"""
import numpy as np

def radial(d, family, scale, amplitude=1.):
    r = np.linalg.norm(d, axis=-1)
    outer = d[..., :, None] * d[..., None, :]
    eye = np.eye(3)
    if family == "rbf":
        k = amplitude * np.exp(-.5 * (r/scale)**2)
        g = -k[...,None] * d / scale**2
        H = k[...,None,None] * (outer/scale**4 - eye/scale**2)
    elif family == "matern32":
        a = np.sqrt(3.) / scale
        e = amplitude * np.exp(-a*r)
        k = (1+a*r)*e
        g = -a*a*e[...,None]*d
        outer_over_r = np.divide(outer, r[...,None,None],
                                out=np.zeros_like(outer), where=r[...,None,None]>0)
        H = -a*a*e[...,None,None]*(eye-a*outer_over_r)
    elif family == "cubic":
        k = r**3
        g = 3*r[...,None]*d
        outer_over_r = np.divide(outer, r[...,None,None],
                                out=np.zeros_like(outer), where=r[...,None,None]>0)
        H = 3*(r[...,None,None]*eye+outer_over_r)
    else:
        raise ValueError("unknown kernel family")
    return k, g, H

def directions(points, normal_vectors, mode):
    if mode == "full":
        return np.broadcast_to(np.eye(3), (len(points),3,3)).copy()
    if mode == "directional":
        return normal_vectors[:,None,:].copy()
    if mode == "value":
        return np.empty((len(points),0,3))
    raise ValueError("mode must be full, directional, or value")

def system(points, dirs, family, scale, amplitude=1.):
    k, g, H = radial(points[:,None,:]-points[None,:,:], family, scale, amplitude)
    n, c = len(points), dirs.shape[1]+1
    K = np.empty((n,c,n,c))
    K[:,0,:,0] = k
    K[:,0,:,1:] = -np.einsum("ijb,jkb->ijk", g, dirs)
    K[:,1:,:,0] = np.einsum("ijb,ikb->ikj", g, dirs)
    K[:,1:,:,1:] = -np.einsum("iab,ijbc,jdc->iajd", dirs, H, dirs)
    return K.reshape(n*c,n*c)

def features(queries, points, dirs, family, scale, amplitude=1.):
    k, g, H = radial(queries[:,None,:]-points[None,:,:], family, scale, amplitude)
    q, n, c = len(queries), len(points), dirs.shape[1]+1
    F = np.empty((q,n,c))
    G = np.empty((q,n,c,3))
    F[:,:,0], G[:,:,0] = k, g
    F[:,:,1:] = -np.einsum("qnb,nab->qna", g, dirs)
    G[:,:,1:] = -np.einsum("qnbc,nac->qnab", H, dirs)
    return F.reshape(q,n*c), G.reshape(q,n*c,3)

def polynomial(points):
    """Observation matrix for the affine basis [1,x,y,z]."""
    P = np.zeros((len(points),4,4))
    P[:,0,0] = 1.
    P[:,0,1:] = points
    P[:,1:,1:] = np.eye(3)
    return P.reshape(-1,4)
