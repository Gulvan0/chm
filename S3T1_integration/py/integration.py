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

def betaZeroInt(x, b, beta):
    return (b-x)**(1 - beta)/(beta-1)

def betaJOrderInt(x0, x1, b, beta, j):
    if j == 0:
        return betaZeroInt(x1, b, beta) - betaZeroInt(x0, b, beta)
    else:
        prevMoment = betaJOrderInt(x0, x1, b, beta, j-1)
        return (x1**j - x0**j - j*b*prevMoment)/(beta - (j + 1))


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
        return [betaJOrderInt(xl, xr, b, beta, s) for s in range(0, max_s + 1)]

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
    m = moments(len(xs)-1, x0, x1, **kwargs)
    q = np.vander(xs)[:, ::-1].T
    a = np.linalg.solve(q, m)

    return a.dot(func(xs))


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
    borders = np.linspace(x0, x1, n_intervals+1)
    s = 0
    for i in range(n_intervals):
        s += quad(func, borders[i], borders[i+1], np.linspace(borders[i], borders[i+1], n_nodes), **kwargs)
    return s


def integrate(func, x0, x1, tol):
    """
    integrate with error <= tol
    return: result, error estimation
    """
    nodes = 3
    L = 2
    h0 = x1-x0
    condReached = False
    while True:
        threeLastSi = []
        h = []
        for i in range(3):
            h.append(h0/L**i)
            n = math.ceil((x1-x0)/h[i])
            threeLastSi.append(composite_quad(func,x0,x1,n, nodes))
        m = aitken(threeLastSi[0], threeLastSi[1], threeLastSi[2], L)
        r1, r2 = runge(threeLastSi[1], threeLastSi[2], m, L)
        if condReached:
            break
        elif r2 < tol:
            h0 *= 0.95
            condReached = True
        else:
            h0 = h[2] * math.pow(tol/abs(r2), 1/m)
    return threeLastSi[2], max(r2, tol)
