"""srccython.pyx - Second order source helper module for cython.

Provides the method interpdps which interpolates results in dp1 and dpdot1.

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import numpy as N
cimport numpy as N
cimport cython

from pyflation.romberg import romb

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
    double cos(double x)

cpdef double klessq2(int kix, int qix, double theta, double kquot):
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
    res = sqrt((kquot + kix)**2 + (kquot + qix)**2 - 2*(kquot + kix)*(kquot + qix)*cos(theta)) - kquot
    return res

@cython.boundscheck(False)    
cpdef interpdps(object dp1_obj,  object dp1dot_obj,
              DTYPEF_t kmin, DTYPEF_t dk, DTYPEI_t kix, 
              object theta_obj,
              DTYPEI_t rmax):
    """Interpolate values of dphi1 and dphi1dot at k=klq.
    
    Parameters
    ----------
    dp1_obj: numpy array
             One dimensional numpy array of dp1 values
             
    dp1dot_obj: numpy array
                One dimensional numpy array of dp1dot values
                
    kmin: float
          minimum k value
          
    dk: float
        difference between two k values
        
    kix: int
         index of current k value
         
    theta_obj: numpy array
               One dimensional numpy array of theta values
               
    rmax: int
          length of k array
          
    Returns
    -------
    dpres: numpy array
           Three dimensional array of shape (2,rmax,tmax) where tmax is length of 
           theta array.
    """
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
            p = klessq2(kix, r, theta[t], kquot)
            if p >= 0.0:
                #get floor and ceiling (cast as ints)
                fp = <int> floor(p)
                cp = <int> ceil(p)
                if fp == cp:
                    pquotient = 0.0
                else:
                    pquotient = (p - fp)/(cp - fp)
                #debug
                
                #Save results
                dpres[0,r,t] = dp1[fp] + pquotient*(dp1[cp]-dp1[fp])
                dpres[1,r,t] = dp1dot[fp] + pquotient*(dp1dot[cp]-dp1dot[fp])                 
            else:
                dpres[0,r,t] = 0
                dpres[1,r,t] = 0
    return dpres
