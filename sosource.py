"""sosource.py Second order source term calculation module.
Author: Ian Huston
$Id: sosource.py,v 1.75 2010/01/18 14:36:02 ith Exp $

Provides the method getsourceandintegrate which uses an instance of a first
order class from cosmomodels to calculate the source term required for second
order models.

"""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0

import numpy as N
from scipy import integrate, interpolate
from romberg import romb
import helpers
import logging
import time
import os
import srccython
import run_config
from run_config import _debug

from pudb import set_trace

if not "profile" in __builtins__:
    def profile(f):
        return f


#This is the results directory which will be used if no filenames are specified
RESULTSDIR = run_config.RESULTSDIR

#Start logging
source_logger = logging.getLogger(__name__)

def set_log_name():
    root_log_name = logging.getLogger().name
    source_logger.name = root_log_name + "." + __name__

def klessq(k, q, theta):
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
    return N.sqrt(k**2+q[..., N.newaxis]**2-2*k*N.outer(q,N.cos(theta)))

@profile
def getthetaterms(integrand_elements, dp1, dp1dot, fullk):
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
#    set_trace()
    k, q, theta = integrand_elements
#    dpshape = [q.shape[0], theta.shape[0]]
#    dpnew = N.array([dp1.real, dp1.imag])
#    dpdnew = N.array([dp1dot.real, dp1dot.imag])
#    dtheta = theta[1]-theta[0]
#    sinth = N.sin(theta)
#    cossinth = N.cos(theta)*N.sin(theta)
    theta_terms = N.empty([4, 2, k.shape[0], q.shape[0]])
    
    dpnew = interpolate.interp1d(fullk, dp1, bounds_error=False, fill_value=N.nan)
    dpdnew = interpolate.interp1d(fullk, dp1dot, bounds_error=False, fill_value=N.nan)
    
    xdpx = lambda x: x*dpnew(x)
    x3dpx = lambda x: x**3 * dpnew(x)
    
    xdpdx = lambda x: x*dpdnew(x)
    x3dpdx = lambda x: x**3 * dpdnew(x)
    
    for n, onek in enumerate(k):
        #klq = klessq(onek, q, theta)
#        dphi_tgther, dphidot_tgther = srccython.interpdps2(dpnew, dpdnew, k[0], k[1]-k[0], n, theta, len(q))
#        for z in range(2):
#            theta_terms[0,z,n] = romb(sinth*dphi_tgther[z], dx=dtheta)
#            theta_terms[1,z,n] = romb(cossinth*dphi_tgther[z], dx=dtheta)
#            theta_terms[2,z,n] = romb(sinth*dphidot_tgther[z], dx=dtheta)
#            theta_terms[3,z,n] = romb(cossinth*dphidot_tgther[z], dx=dtheta)
        for r, oneq in enumerate(q):
            lower_limit = N.abs(onek-oneq)
            upper_limit = onek+oneq
            dpint1 = integrate.romberg(xdpx, lower_limit, upper_limit, vec_func=True)
            dpres1 = dpint1/(onek*oneq)
            theta_terms[0,:,n,r] = dpres1.real, dpres1.imag
            dpint2 = integrate.romberg(x3dpx, lower_limit, upper_limit, vec_func=True)
            dpres2 = (-dpint2 + (onek**2+oneq**2)*dpint1)/(2*(onek*oneq)**2)
            theta_terms[1,:,n,r] = dpres2.real, dpres2.imag
            dpdint1 = integrate.romberg(xdpdx, lower_limit, upper_limit, vec_func=True)
            dpdres1 = dpdint1/(onek*oneq)
            theta_terms[2,:,n,r] = dpdres1.real, dpdres1.imag
            dpdint2 = integrate.romberg(x3dpdx, lower_limit, upper_limit, vec_func=True)
            dpdres2 = (-dpdint2 + (onek**2+oneq**2)*dpdint1)/(2*(onek*oneq)**2)
            theta_terms[3,:,n,r] = dpdres2.real, dpdres2.imag
            
    
    return theta_terms

        
def slowrollsrcterm(bgvars, a, potentials, integrand_elements, dp1, dp1dot, theta_terms):
    """Return unintegrated slow roll source term.
    
    The source term before integration is calculated here using the slow roll
    approximation. This function follows the revised version of Eq (5.8) in 
    Malik 06 (astro-ph/0610864v5).
    
    Parameters
    ----------
    bgvars: tuple
            Tuple of background field values in the form `(phi, phidot, H)`
    
    a: float
       Scale factor at the current timestep, `a = ainit*exp(n)`
    
    potentials: tuple
                Tuple of potential values in the form `(U, dU, dU2, dU3)`
    
    integrand_elements: tuple 
         Contains integrand arrays in order (k, q, theta)
            
    dp1: array_like
         Array of known dp1 values
             
    dp1dot: array_like
            Array of dpdot1 values
             
    theta_terms: array_like
                 3-d array of integrated theta terms of shape (4, len(k), len(q))
             
    Returns
    -------
    src_integrand: array_like
        Array containing the unintegrated source terms for all k and q modes.
        
    References
    ----------
    Malik, K. 2006, JCAP03(2007)004, astro-ph/0610864v5
    """
    #Unpack variables
    phi, phidot, H = bgvars
    U, dU, dU2, dU3 = potentials
    k = integrand_elements[0][...,N.newaxis]
    q = integrand_elements[1][N.newaxis, ...]
    #Calculate dphi(q) and dphi(k-q)
    dp1_q = dp1[N.newaxis,:q.shape[-1]]
    dp1dot_q = dp1dot[N.newaxis,q.shape[-1]]
    atmp, btmp, ctmp, dtmp = theta_terms
    aterm = atmp[0] + atmp[1]*1j
    bterm = btmp[0] + btmp[1]*1j
    cterm = ctmp[0] + ctmp[1]*1j
    dterm = dtmp[0] + dtmp[1]*1j
    
    #Calculate unintegrated source term
    #First major term:
    if dU3!=0.0:
        src_integrand = ((1/H**2) * dU3 * q**2 * dp1_q * aterm)
    else:
        src_integrand = N.zeros_like(aterm)
    #Second major term:
    src_integrand2 = (phidot/((a*H)**2)) * ((3*dU2*(a*q)**2 + 3.5*q**4 + 2*(k**2)*(q**2))*aterm 
                      - (4.5 + (q/k)**2)* k * (q**3) * bterm) * dp1_q
    #Third major term:
    src_integrand3 = (phidot * ((-1.5*q**2)*cterm + (2 - (q/k)**2)*k*q*dterm) * dp1dot_q)
    #Multiply by prefactor
    src_integrand = src_integrand + src_integrand2 + src_integrand3
    src_integrand *= (1/(2*N.pi)**2)
    
    return src_integrand

def calculatesource(m, nix, integrand_elements, srcfunc=slowrollsrcterm):
    """Return the integrated source term at this timestep.
    
    Given the first order model and the timestep calculate the 
    integrated source term at this time step.
    
    Parameters
    ----------
    m: Cosmomodels.TwoStageModel
       First order model to be used.
       
    nix: int
         Index of current timestep in m.tresult
     
    integrand_elements: tuple
         Contains integrand arrays in order
         integrand_elements = (k, q, theta)
         
    srcfunc: function, optional
             Funtion which contains expression for source integrand
             Default is slowrollsrcterm in this module.        
         
    Returns
    -------
    s: array_like
       Integrated source term calculated using srcfunc.
    """
    #Copy of yresult
    myr = m.yresult[nix].copy()
    #Fill nans with initial conditions
    if N.any(m.tresult[nix] < m.fotstart):
        #Get first order ICs:
        nanfiller = m.getfoystart(m.tresult[nix].copy(), N.array([nix]))
        if _debug:
            source_logger.debug("Left getfoystart. Filling nans...")
        #switch nans for ICs in m.yresult
        are_nan = N.isnan(myr)
        myr[are_nan] = nanfiller[are_nan]
        if _debug:
            source_logger.debug("NaNs filled. Setting dynamical variables...")
    #Get first order results
    bgvars = myr[0:3,0]
    dphi1 = myr[3,:] + myr[5,:]*1j
    dphi1dot = myr[4,:] + myr[6,:]*1j
    #Setup interpolation
    if _debug:
        source_logger.debug("Variables set. Getting potentials for this timestep...")
    potentials = list(m.potentials(myr))
    #Get potentials in right shape
    for pix, p in enumerate(potentials):
        if N.shape(p) != N.shape(potentials[3]):
            potentials[pix] = p[0]
    #Value of a for this time step
    a = m.ainit*N.exp(m.tresult[nix])
    if _debug:
        source_logger.debug("Calculating source term integrand for this timestep...")
    theta_terms = getthetaterms(integrand_elements, dphi1, dphi1dot, m.k)
    #Get unintegrated source term
    src_integrand = srcfunc(bgvars, a, potentials, integrand_elements, dphi1, dphi1dot, theta_terms)
    #Get integration function
    if _debug:
        source_logger.debug("Integrating source term...")
        source_logger.debug("Number of integrand elements: %f", src_integrand.shape[-1])
    src = integrate.romb(src_integrand, dx=m.k[1]-m.k[0])
    
    
    if _debug:
        source_logger.debug("Integration successful!")
    return src

@profile                
def getsourceandintegrate(m, savefile=None, srcfunc=slowrollsrcterm, ninit=0, nfinal=-1, ntheta=513, numks=1025):
    """Calculate and save integrated source term.
    
    Using first order results in the specified model, the source term for second order perturbations 
    is obtained from the given source function. The convolution integral is performed and the results
    are saved in a file with the specified filename.
    
    Parameters
    ----------
    m: compatible cosmomodels model instance
       The model class should contain first order results as in `cosmomodels.FOCanonicalTwoStage`
    
    savefile: String, optional
              Filename where results should be saved.
    
    srcfunc: function, optional
             Function which returns unintegrated source term. Defaults to slowrollsrcterm in this module.
             Function signature is `srcfunc(bgvars, a, potentials, integrand_elements, dp1func, dp1dotfunc)`.
             
    ninit: int, optional
           Start time index for source calculation. Default is 0 (start at beginning).
    
    nfinal: int, optional
            End time index for source calculation. -1 signifies end of run (and is the default).
    
    ntheta: int, optional
            Number of theta points to integrate. Should be a power of two + 1 for Romberg integration.
            Default is 129.
    
    numks: int, optional
           Number of k modes required for second order calculation. Must be small enough that largest first
           order k mode is at least 2*(numks*deltak + kmin). Default is this limiting value.
               
    Returns
    -------
    savefile: String
              Filename where results have been saved.
    """
    #Check time limits
    if ninit < 0:
        ninit = 0
    if nfinal > m.tresult.shape[0] or nfinal == -1:
        nfinal = m.tresult.shape[0]
    #Change to ints for xrange
    ninit = int(ninit)
    nfinal = int(nfinal)    
    
    if not numks:
        numks = round((m.k[-1]/2 - m.k[0])/(m.k[1]-m.k[0])) + 1
    #Initialize variables for all timesteps
    k = q = m.k[:numks] #Need N=len(m.k)/2 for case when q=-k
    #Check consistency of first order k range
    if (m.k[-1] - 2*k[-1])/m.k[-1] < -1e-12:
        raise ValueError("First order k range not sufficient!")
    theta = N.linspace(0, N.pi, ntheta)
    firstmodestart = min(m.fotstart)
    lastmodestart = max(m.fotstart)
   
    #Pack together in tuple
    integrand_elements = (k, q, theta)
    
    #Main try block for file IO
    try:
        sf, sarr, narr = opensourcefile(k, savefile, sourcetype="term")
        try:
            # Begin calculation
            if _debug:
                source_logger.debug("Entering main time loop...")    
            #Main loop over each time step
            for nix in xrange(ninit, nfinal):    
                if nix%10 == 0:
                    source_logger.info("Starting nix=%d/%d sequence...", nix, nfinal)
                #Only run calculation if one of the modes has started.
                if m.tresult[nix] > lastmodestart or m.tresult[nix+2] >= firstmodestart:
                    src = calculatesource(m, nix, integrand_elements, srcfunc)
                else:
                    src = N.nan*N.ones_like(k)
                sarr.append(src[N.newaxis,:])
                narr.append(N.array([nix]))
                if _debug:
                    source_logger.debug("Results for this timestep saved.")
        finally:
            #source = N.array(source)
            sf.close()
    except IOError:
        raise
    return savefile

def opensourcefile(k, filename=None, sourcetype=None):
    """Open the source term hdf5 file with filename."""
    import tables
    #Set up file for results
    if not filename or not os.path.isdir(os.path.dirname(filename)):
        source_logger.info("File or path to file %s does not exist." % filename)
        date = time.strftime("%Y%m%d%H%M%S")
        filename = RESULTSDIR + "src" + date + ".hf5"
        source_logger.info("Saving source results in file " + filename)
    if not sourcetype:
        raise TypeError("Need to specify filename and type of source data to store [int(egrand)|(full)term]!")
    if sourcetype in ["int", "term"]:
        sarrname = "source" + sourcetype
        if _debug:
            source_logger.debug("Source array type: " + sarrname)
    else:
        raise TypeError("Incorrect source type specified!")
    #Add compression to files and specify good chunkshape
    filters = tables.Filters(complevel=1, complib="zlib") 
    #cshape = (10,10,10) #good mix of t, k, q values
    #Get atom shape for earray
    atomshape = (0, len(k))
    try:
        if _debug:
            source_logger.debug("Trying to open source file " + filename)
        rf = tables.openFile(filename, "a", "Source term result")
        if not "results" in rf.root:
            if _debug:
                source_logger.debug("Creating group 'results' in source file.")
            resgrp = rf.createGroup(rf.root, "results", "Results")
        else:
            resgrp = rf.root.results
        if not sarrname in resgrp:
            if _debug:
                source_logger.debug("Creating array '" + sarrname + "' in source file.")
            sarr = rf.createEArray(resgrp, sarrname, tables.ComplexAtom(itemsize=16), atomshape, filters=filters)
            karr = rf.createEArray(resgrp, "k", tables.Float64Atom(), (0,), filters=filters)
            narr = rf.createEArray(resgrp, "nix", tables.IntAtom(), (0,), filters=filters)
            karr.append(k)
        else:
            if _debug:
                source_logger.debug("Source file and node exist. Testing source node shape...")
            sarr = rf.getNode(resgrp, sarrname)
            narr = rf.getNode(resgrp, "nix")
            if sarr.shape[1:] != atomshape[1:]:
                raise ValueError("Source node on file is not correct shape!")
    except IOError:
        raise
    return rf, sarr, narr
