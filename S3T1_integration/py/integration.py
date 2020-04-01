import numpy as np
import math

def i0(x, a, alpha):
    return (x-a)**(1 - alpha)/(1 - alpha)

def ij(x, a, alpha, j):
    if j == 0:
        return i0(x, a, alpha)
    else:
        ijminus1 = ij(x, a, alpha, j-1)
    return (ijminus1*j*a + (x-a)**(1 - alpha) * x**j)/(j+ 1 - alpha)

def i0b(x, b, beta):
    return (b-x)**(1 - beta)/(beta-1)

def ijb(x, b, beta, j):
    if j == 0:
        return i0b(x, b, beta)
    else:
        ijminus1 = ijb(x, b, beta, j-1)
    return (ijminus1*j*b + (-1)**(j+1) * (b-x)**(1 - beta) * x**j)/(j+ 1 - beta)


def moments(max_s, xl, xr, a=None, b=None, alpha=0.0, beta=0.0):
    """
    compute 0..max_s moments of the weight p(x) = 1 / (x-a)^alpha / (b-x)^beta over [xl, xr]
    """
    assert alpha * beta == 0, f'alpha ({alpha}) and/or beta ({beta}) must be 0'
    if alpha != 0.0:
        assert a is not None, f'"a" not specified while alpha != 0'
        return [ij(xr,a, alpha,s) - ij(xl,a, alpha,s) for s in range(0, max_s + 1)]
    if beta != 0.0:
        assert b is not None, f'"b" not specified while beta != 0'
        return [(ijb(xr,b, beta,s) - ijb(xl,b, beta,s)) for s in range(0, max_s + 1)]

    if alpha == 0 and beta == 0:
        return [(xr ** s - xl ** s) / s for s in range(1, max_s + 2)]

    raise NotImplementedError


def runge(s0, s1, m, L):
    """
    estimate m-degree errors for s0 and s1
    """
    d0 = np.abs(s1 - s0) / (1 - L ** -m)
    d1 = np.abs(s1 - s0) / (L ** m - 1)
    return d0, d1


def aitken(s0, s1, s2, L):
    """
    estimate accuracy degree
    s0, s1, s2: consecutive composite quads
    return: accuracy degree estimation
    """
    return - (math.log(abs((s2 - s1)/(s1 - s0)))/math.log(L))


def quad(func, x0, x1, xs, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    xs: nodes
    **kwargs passed to moments()
    """
    q = [[p**s for p in xs] for s in range(len(xs))]
    return np.sum(np.linalg.solve(q, moments(len(xs)-1, x0, x1, **kwargs)).dot(func(xs)))


def quad_gauss(func, x0, x1, n, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n: number of nodes
    """
    moms = moments(2*n-1, x0, x1, **kwargs)
    q = [[moms[i+j] for i in range(n)] for j in range(n)]
    b = [-moms[i+n] for i in range(n)]
    a = np.linalg.solve(q, b)
    xs = np.roots(np.append(a, 1)[::-1])
    q2 = [[p**s for p in xs] for s in range(n)]
    return np.sum(np.linalg.solve(q2, moms[:n]).dot(func(xs)))


def composite_quad(func, x0, x1, n_intervals, n_nodes, **kwargs):
    """
    func: function to integrate
    x0, x1: interval to integrate on
    n_intervals: number of intervals
    n_nodes: number of nodes on each interval
    """
    borders = [x0 + i * (x1 - x0) / n_intervals for i in range(n_intervals+1)]
    s = 0
    for i in range(n_intervals):
        s += quad_gauss(func, borders[i], borders[i+1], n_nodes, **kwargs)
    return s


def integrate(func, x0, x1, tol):
    """
    integrate with error <= tol
    return: result, error estimation
    """
    raise NotImplementedError
