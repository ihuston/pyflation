'''romberg.py
Created on 14 Jul 2010

Adapted from scipy.integrate by Ian Huston

'''

from numpy import add, isscalar, asarray

if not "profile" in __builtins__:
    def profile(f):
        return f

def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)

@profile
def romb(y, dx=1.0, axis=-1, show=False):
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
    y = asarray(y)
    nd = len(y.shape)
    Nsamps = y.shape[axis]
    Ninterv = Nsamps-1
    n = 1
    k = 0
    while n < Ninterv:
        n <<= 1
        k += 1
    if n != Ninterv:
        raise ValueError, \
              "Number of samples must be one plus a non-negative power of 2."

    R = {}
    all = (slice(None),) * nd
    slice0 = tupleset(all, axis, 0)
    slicem1 = tupleset(all, axis, -1)
    h = Ninterv*asarray(dx)*1.0
    R[(1,1)] = (y[slice0] + y[slicem1])/2.0*h
    slice_R = all
    start = stop = step = Ninterv
    for i in range(2,k+1):
        start >>= 1
        slice_R = tupleset(slice_R, axis, slice(start,stop,step))
        step >>= 1
        R[(i,1)] = 0.5*(R[(i-1,1)] + h*add.reduce(y[slice_R],axis))
        for j in range(2,i+1):
            R[(i,j)] = R[(i,j-1)] + \
                       (R[(i,j-1)]-R[(i-1,j-1)]) / ((1 << (2*(j-1)))-1)
        h = h / 2.0

    if show:
        if not isscalar(R[(1,1)]):
            print "*** Printing table only supported for integrals" + \
                  " of a single data set."
        else:
            try:
                precis = show[0]
            except (TypeError, IndexError):
                precis = 5
            try:
                width = show[1]
            except (TypeError, IndexError):
                width = 8
            formstr = "%" + str(width) + '.' + str(precis)+'f'

            print "\n       Richardson Extrapolation Table for Romberg Integration       "
            print "===================================================================="
            for i in range(1,k+1):
                for j in range(1,i+1):
                    print formstr % R[(i,j)],
                print
            print "====================================================================\n"

    return R[(k,k)]
