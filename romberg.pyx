'''romberg.py
Created on 14 Jul 2010

Adapted from scipy.integrate by Ian Huston

'''

from numpy import add, isscalar, asarray

import numpy as N
cimport numpy as N
cimport cython

DTYPED = N.float64
DTYPEI = N.int
ctypedef N.double_t DTYPED_t
ctypedef N.int_t DTYPEI_t
DTYPEC = N.complex128
ctypedef N.complex128_t DTYPEC_t



def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


def romb(N.ndarray[DTYPEC_t, ndim=2] y, DTYPED_t dx=1.0):
    """Romberg integration using samples of a function

    Inputs:

       y    -  a vector of 2**k + 1 equally-spaced samples of a function
       dx   -  the sample spacing.
       axis -  the axis along which to integrate
       show -  When y is a single 1-d array, then if this argument is True
               print the table showing Richardson extrapolation from the
               samples.

    Output: ret

       ret  - The integrated result for each axis.

    See also:

      quad - adaptive quadrature using QUADPACK
      romberg - adaptive Romberg quadrature
      quadrature - adaptive Gaussian quadrature
      fixed_quad - fixed-order Gaussian quadrature
      dblquad, tplquad - double and triple integrals
      simps, trapz - integrators for sampled data
      cumtrapz - cumulative integration for sampled data
      ode, odeint - ODE integrators
    """
#    y = asarray(y)
    cdef DTYPEI_t nd = 2
    cdef DTYPEI_t Nsamps = y.shape[1]
    cdef DTYPEI_t Ninterv = Nsamps-1
    cdef DTYPEI_t n = 1
    cdef DTYPEI_t k = 0
    cdef int i, j
    cdef DTYPEI_t start, stop, step
    
    while n < Ninterv:
        n <<= 1
        k += 1
    if n != Ninterv:
        raise ValueError, \
              "Number of samples must be one plus a non-negative power of 2."

    R = {}
    all = (slice(None),) * nd
    slice0 = tupleset(all, 1, 0)
    slicem1 = tupleset(all, 1, -1)
    h = Ninterv*asarray(dx)*1.0
    R[(1,1)] = (y[slice0] + y[slicem1])/2.0*h
    slice_R = all
    start = stop = step = Ninterv
    for i in range(2,k+1):
        start >>= 1
        slice_R = tupleset(slice_R, 1, slice(start,stop,step))
        step >>= 1
        R[(i,1)] = 0.5*(R[(i-1,1)] + h*add.reduce(y[slice_R],1))
        for j in range(2,i+1):
            R[(i,j)] = R[(i,j-1)] + \
                       (R[(i,j-1)]-R[(i-1,j-1)]) / ((1 << (2*(j-1)))-1)
        h = h / 2.0

    return R[(k,k)]
