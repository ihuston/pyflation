"""srccython.pyx Second order source helper module for cython.
Author: Ian Huston

$Id: srccython.pyx,v 1.2 2009/02/27 17:45:21 ith Exp $
Provides the method interpdps which interpolates results in dp1 and dpdot1.

"""



from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N
cimport numpy as N
cimport cython
from romberg import romb

DTYPEF = N.float
DTYPEI = N.int
ctypedef N.float_t DTYPEF_t
ctypedef N.int_t DTYPEI_t
DTYPEC = N.complex128
ctypedef N.complex128_t DTYPEC_t


cdef extern from "math.h":
    double sqrt(double x)
    double ceil(double x)
    double floor(double x)

cdef double klessq2(int kix, int qix, double theta, double kmin, double kquot):
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
    cdef double res
    res = sqrt((kquot + kix)**2 + (kquot + qix)**2 - 2*(kquot + kix)*(kquot + qix)*theta) - kmin
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

@cython.boundscheck(False)    
cpdef interpdps2(object dp1_obj,  object dp1dot_obj,
              DTYPEF_t kmin, DTYPEF_t dk, DTYPEI_t kix, 
              object theta_obj,
              DTYPEI_t rmax):
    """Interpolate values of dphi1 and dphi1dot at k=klq."""
    cdef N.ndarray[DTYPEC_t, ndim=1] dp1 = dp1_obj
    cdef N.ndarray[DTYPEC_t, ndim=1] dp1dot = dp1dot_obj
    cdef N.ndarray[DTYPEF_t, ndim=1] theta = theta_obj
    #cdef N.ndarray[DTYPEF_t, ndim=2] klqix = (klq - kmin)/dk #Supposed indices
    #cdef int rmax = klq.shape[0]
    cdef int tmax = theta.shape[0]
    cdef double kquot = kmin/dk
    cdef int r, t, z
    cdef double p, pquotient
    cdef int fp, cp
    cdef N.ndarray[DTYPEC_t, ndim=3] dpres = N.empty((2,rmax,tmax), dtype=DTYPEC)
    
    for r in range(rmax):
        for t in range(tmax):
            p = klessq2(kix, r, theta[t], kmin, kquot)
            if p >= 0.0:
                #get floor and ceiling (cast as ints)
                fp = <int> floor(p)
                cp = <int> ceil(p)
                if fp == cp:
                    pquotient = 0.0
                else:
                    pquotient = (p - fp)/(cp - fp)
                #Save results
                dpres[0,r,t] = dp1[fp] + pquotient*(dp1[cp]-dp1[fp])
                dpres[1,r,t] = dp1dot[fp] + pquotient*(dp1dot[cp]-dp1dot[fp])
    return dpres


def getthetaterms(N.ndarray[DTYPEF_t, ndim=1] k, N.ndarray[DTYPEF_t, ndim=1] q,
                  N.ndarray[DTYPEF_t, ndim=1] theta, 
                  N.ndarray[DTYPEC_t, ndim=1] dp1, 
                  N.ndarray[DTYPEC_t, ndim=1] dp1dot):
    """Return array of integrated values for specified theta function and dphi function.
    
    Parameters
    ----------
    integrand_elements: tuple
            Contains integrand arrays in order (k, q, theta)
             
    dp1: array_like
         Array of values for dphi1
    
    dp1dot: array_like
            Array of values for dphi1dot
                                  
    Returns
    -------
    theta_terms: tuple
                 Tuple of len(k)xlen(q) shaped arrays of integration results in form
                 (\int(sin(theta) dp1(k-q) dtheta,
                  \int(cos(theta)sin(theta) dp1(k-q) dtheta,
                  \int(sin(theta) dp1dot(k-q) dtheta,
                  \int(cos(theta)sin(theta) dp1dot(k-q) dtheta)
                 
    """
    #k, q, theta = integrand_elements
    #dpshape = [q.shape[0], theta.shape[0]]
#    dpnew = N.array([dp1.real, dp1.imag])
#    dpdnew = N.array([dp1dot.real, dp1dot.imag])
    cdef DTYPEF_t dtheta = theta[1]-theta[0]
    cdef N.ndarray[DTYPEF_t, ndim=1] sinth = N.sin(theta)
    cdef N.ndarray[DTYPEF_t, ndim=1] cossinth = N.cos(theta)*N.sin(theta)
    
    cdef int lenk = k.shape[0]
    cdef int lenq = q.shape[0]
    cdef N.ndarray[DTYPEC_t, ndim=3] theta_terms = N.empty([4, lenk, lenq], dtype=DTYPEC)
    cdef N.ndarray[DTYPEC_t, ndim=3] dphi_res = N.empty((2,lenq,theta.shape[0]), dtype=DTYPEC)
    
    
    cdef DTYPEF_t dk = k[1]-k[0]
    cdef DTYPEF_t kmin = k[0]
    for n in range(lenk):
        #klq = klessq(onek, q, theta)
        dphi_res = interpdps2(dp1, dp1dot, kmin, dk, n, theta, lenq)
        
        theta_terms[0,n] = romb(sinth*dphi_res[0], dx=dtheta)
        theta_terms[1,n] = romb(cossinth*dphi_res[0], dx=dtheta)
        theta_terms[2,n] = romb(sinth*dphi_res[1], dx=dtheta)
        theta_terms[3,n] = romb(cossinth*dphi_res[1], dx=dtheta)
    return theta_terms