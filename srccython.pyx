"""srccython.pyx Second order source helper module for cython.
Author: Ian Huston

$Id: srccython.pyx,v 1.2 2009/02/27 17:45:21 ith Exp $
Provides the method interpdps which interpolates results in dp1 and dpdot1.

"""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N
cimport numpy as N
DTYPEF = N.float
DTYPEI = N.int
ctypedef N.float_t DTYPEF_t
ctypedef N.int_t DTYPEI_t

from scipy import integrate

cdef extern from "math.h":
    double sqrt(double x)
    double ceil(double x)
    double floor(double x)

def klessq(double k, N.ndarray[DTYPEF_t] q, N.ndarray[DTYPEF_t] theta):
    """Return the scalar magnitude of k^i - q^i where theta is angle between vectors.
    
    Parameters
    ----------
    k: float
       Single k value to compute array for.
    
    q: array_like
       1-d array of q values to use
     
    theta: array_like
           1-d array of theta values to use
           
    Returns
    -------
    klessq: array_like
            len(q)*len(theta) array of values for
            |k^i - q^i| = \sqrt(k^2 + q^2 - 2kq cos(theta))
    """
#     N.sqrt(k**2+q[..., N.newaxis]**2-2*k*N.outer(q,N.cos(theta)))
    
    cdef int r, t
    cdef int rmax = q.shape[0]
    cdef int tmax = theta.shape[0]
    cdef N.ndarray[DTYPEF_t, ndim=2] res = N.empty([rmax, tmax], dtype=DTYPEF)
    
    for r in range(rmax):
        for t in range(tmax):
            res[r,t] = sqrt(k**2 + q[r]**2 - 2*k*q[r]*theta[t])
    return res

def interpdps(N.ndarray[DTYPEF_t, ndim=2] dp1, N.ndarray[DTYPEF_t, ndim=2] dp1dot,
              DTYPEF_t kmin, DTYPEF_t dk, N.ndarray[DTYPEF_t, ndim=2] klq):
    """Interpolate values of dphi1 and dphi1dot at k=klq."""
    cdef N.ndarray[DTYPEF_t, ndim=2] klqix = (klq - kmin)/dk #Supposed indices
    cdef int rmax = klq.shape[0]
    cdef int tmax = klq.shape[1]
    cdef int r, t, z
    cdef float p, pquotient
    cdef int fp, cp
    cdef N.ndarray[DTYPEF_t, ndim=4] dpres = N.zeros((2,2,rmax,tmax))
    for r in range(rmax):
        for t in range(tmax):
            p = klqix[r,t]
            if p >= 0.0:
                #get floor and ceiling (cast as ints)
                fp = <int> floor(p)
                cp = <int> ceil(p)
                if fp == cp:
                    pquotient = 0.0
                else:
                    pquotient = (p - fp)/(cp - fp)
                #Save results
                for z in range(2):
                    dpres[0,z,r,t] = dp1[z,fp] + pquotient*(dp1[z,cp]-dp1[z,fp])
                    dpres[1,z,r,t] = dp1dot[z,fp] + pquotient*(dp1dot[z,cp]-dp1dot[z,fp])
    return dpres

def getthetaterms(integrand_elements, dp1func, dp1dotfunc):
    """Return array of integrated values for specified theta function and dphi function.
    
    Parameters
    ----------
    integrand_elements: tuple
            Contains integrand arrays in order (k, q, theta)
             
    dp1func: function object
             Function of klessq for dphi1 e.g. interpolated function of dphi1 results.
    
    dp1dotfunc: function object
             Function of klessq for dphi1dot e.g. interpolated function of dphi1dot results.
                                  
    Returns
    -------
    theta_terms: tuple
                 Tuple of len(k)xlen(q) shaped arrays of integration results in form
                 (\int(sin(theta) dp1(k-q) dtheta,
                  \int(cos(theta)sin(theta) dp1(k-q) dtheta,
                  \int(sin(theta) dp1dot(k-q) dtheta,
                  \int(cos(theta)sin(theta) dp1dot(k-q) dtheta)
                 
    """
    k, q, theta = integrand_elements
    dpshape = [q.shape[0], theta.shape[0]]
    dtheta = theta[1]-theta[0]
    sinth = N.sin(theta)
    cossinth = N.cos(theta)*N.sin(theta)
    aterm, bterm, cterm, dterm = [],[],[],[] #Results lists
    for onek in k:
        klq = klessq(onek, q, theta)
        dphi_klq = N.empty(dpshape)
        dphidot_klq = N.empty(dpshape)
        for r in xrange(len(q)):
            oklq = klq[r]
            dphi_klq[r] = dp1func(oklq)
            dphidot_klq[r] = dp1dotfunc(oklq)
        aterm.append(integrate.romb(sinth*dphi_klq, dtheta))
        bterm.append(integrate.romb(cossinth*dphi_klq, dtheta))
        cterm.append(integrate.romb(sinth*dphidot_klq, dtheta))
        dterm.append(integrate.romb(cossinth*dphidot_klq, dtheta))
    theta_terms = N.array(aterm), N.array(bterm), N.array(cterm), N.array(dterm)
    return theta_terms   