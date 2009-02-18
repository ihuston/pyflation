"""sosource.py Second order source term calculation module.
Author: Ian Huston
$Id: sosource.py,v 1.50 2009/02/18 11:54:23 ith Exp $

Provides the method getsourceandintegrate which uses an instance of a first
order class from cosmomodels to calculate the source term required for second
order models.

"""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import tables
import numpy as N
from scipy import integrate
import helpers
import logging
import time
import os

#psyco
import psyco
psyco.full()

#This is the results directory which will be used if no filenames are specified
RESULTSDIR = "/misc/scratch/ith/numerics/results/"

#Start logging
source_logger = logging.getLogger(__name__)

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

def interpdps(dp1, dp1dot, kmin, klq):
    """Interpolate values of dphi1 and dphi1dot at k=klq."""
    klqix = klq/kmin - 1.0 #Supposed indices
    rmax = klq.shape[0]
    tmax = klq.shape[1]
    dpres = N.empty([2, rmax, tmax])
    
    for r in range(rmax):
        for t in range(tmax):
            p = klqix[r,t]
            if p < 0:
                #This is the cut out area of the integral
                dpres[0,r,t] = 0.0
                dpres[1,r,t] = 0.0
            else:
                #get floor and ceiling (cast as ints)
                fp = N.floor(p)
                cp = N.ceil(p)
                pquotient = (p - fp)/(cp - fp)
                #Save results
                dpres[0,r,t] = dp1[fp] + pquotient*(dp1[cp]-dp1[fp])
                dpres[1,r,t] = dp1dot[fp] + pquotient*(dp1dot[cp]-dp1dot[fp])
    return dpres
            
def getthetaterms(k, q, theta, dp1, dp1dot):
    """Return array of integrated values for specified theta function and dphi function.
    
    Parameters
    ----------
    k: array_like
       Array of k values
       
    q: array_like
       Array of q values
    
    theta: array_like
           Array of theta values
             
    dp1: array_like
         Array of calculated dphi1 values for this timestep.
    
    dp1dot: array_like
            Array of calculated dphi1dot values for this timestep.
                                  
    Returns
    -------
    theta_terms: tuple
                 Tuple of len(k)xlen(q) shaped arrays of integration results in form
                 (\int(sin(theta) dp1(k-q) dtheta,
                  \int(cos(theta)sin(theta) dp1(k-q) dtheta,
                  \int(sin(theta) dp1dot(k-q) dtheta,
                  \int(cos(theta)sin(theta) dp1dot(k-q) dtheta)
                 
    """
    dtheta = theta[1]-theta[0]
    sinth = N.sin(theta)
    cossinth = N.cos(theta)*N.sin(theta)
    #indexing variables
    nmax, rmax, tmax = k.shape[0], q.shape[0], theta.shape[0]
    #Results arrays
    theta_terms = N.empty([4, nmax, rmax]) #4terms in array
    
    for n, onek in enumerate(k):
        klq = klessq(onek, q, theta)
        dphi_klq, dphidot_klq = interpdps(dp1, dp1dot, k[0], klq)
        theta_terms[0,n] = integrate.romb(sinth*dphi_klq, dtheta)
        theta_terms[1,n] = integrate.romb(cossinth*dphi_klq, dtheta)
        theta_terms[2,n] = integrate.romb(sinth*dphidot_klq, dtheta)
        theta_terms[3,n] = integrate.romb(cossinth*dphidot_klq, dtheta)
    return theta_terms
        
def slowrollsrcterm(bgvars, a, potentials, integrand_elements, dp1func, dp1dotfunc):
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
            
    dp1func: function object
             Interpolation function for \delta\phi_1
             
    dp1dotfunc: function object
             Interpolation function for \dot{\delta\phi_1}
             
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
    k, q, theta, klessq = integrand_elements
    #Calculate dphi(q) and dphi(k-q)
    dp1_q = dp1func(q)
    dp1_klessq = dp1func(klessq)
    dp1dot_q = dp1dotfunc(q)
    dp1dot_klessq = dp1dotfunc(klessq)
                
    #Calculate unintegrated source term
    #First major term:
    s2 = ((1/H**2) * (dU3 + 3*phidot*dU2) * q**2*dp1diff*dphi1)
    #Second major term:
    s2 += ((-1.5*q**2) * phidot * dp1dotdiff * dphi1dot)
    #Third major term:
    s2 += (1/((a*H)**2) * (2.5*q**4 + 2*(k*q)**2) * phidot * dp1diff * dphi1)
    #Multiply by prefactor
    s2 = s2 * (1/(2*N.pi**2))
    
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
    #Unpack important arrays
    k, q, theta = integrand_elements
    
    #Copy of yresult
    myr = m.yresult[nix].copy()
    #Fill nans with initial conditions
    if N.any(m.tresult[nix] < m.fotstart):
        #Get first order ICs:
        nanfiller = m.getfoystart(m.tresult[nix].copy(), N.array([nix]))
        source_logger.debug("Left getfoystart. Filling nans...")
        #switch nans for ICs in m.yresult
        are_nan = N.isnan(myr)
        myr[are_nan] = nanfiller[are_nan]
        source_logger.debug("NaNs filled. Setting dynamical variables...")
    #Get first order results
    bgvars = myr[0:3,:]
    dphi1 = myr[3,:] + myr[4,:]*1j
    dphi1dot = myr[5,:] + myr[6,:]*1j
    
    source_logger.debug("Variables set. Getting potentials for this timestep...")
    potentials = list(m.potentials(myr))
    #Get potentials in right shape
    for pix, p in enumerate(potentials):
        if N.shape(p) != N.shape(potentials[0]):
            potentials[pix] = p*N.ones_like(potentials[0])
    #Value of a for this time step
    a = m.ainit*N.exp(m.tresult[nix])
    source_logger.debug("Getting theta terms...")
    thetaterms = getthetaterms(k, q, theta, dphi1, dphi1dot)
    source_logger.debug("Calculating source term integrand for this timestep...")
    #Get unintegrated source term
    src_integrand = srcfunc(bgvars, a, potentials, integrand_elements, dp1func, dp1dotfunc)
    #Get integration function
    source_logger.debug("Integrating source term...")

    source_logger.debug("Integration successful!")
    return s3
                
def getsourceandintegrate(m, savefile=None, srcfunc=slowrollsrcterm, ntheta=129):
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
               
    Returns
    -------
    savefile: String
              Filename where results have been saved.
    """
    #Initialize variables for all timesteps
    k = q = m.k[:len(m.k)/2] #Need N=len(m.k)/2 for case when q=-k
    theta = N.linspace(0, N.pi, ntheta)
   
    #Pack together in tuple
    integrand_elements = (k, q, theta)
    
    #Main try block for file IO
    try:
        sf, sarr = opensourcefile(atomshape, halfk, savefile, sourcetype="term")
        try:
            # Begin calculation
            source_logger.debug("Entering main time loop...")    
            #Main loop over each time step
            for nix in xrange(len(m.tresult)):    
                if nix%1000 == 0:
                    source_logger.info("Starting n=%f, nix=%d sequence...", m.tresult[nix], nix)
                #Only run calculation if one of the modes has started.
                if any(m.tresult[nix+2] >= m.fotstart):
                    src = calculatesource(m, nix, integrand_elements, srcfunc)
                else:
                    src = N.nan*N.ones_like(m.k)
                sarr.append(src[N.newaxis,:])
                source_logger.debug("Results for this timestep saved.")
        finally:
            #source = N.array(source)
            sf.close()
    except IOError:
        raise
    return savefile

def opensourcefile(atomshape, k, filename=None, sourcetype=None):
    """Open the source term hdf5 file with filename."""
    #Set up file for results
    if not filename or not os.path.isdir(os.path.dirname(filename)):
        date = time.strftime("%Y%m%d%H%M%S")
        filename = RESULTSDIR + "src" + date + ".hf5"
        source_logger.info("Saving source results in file " + filename)
    if not sourcetype:
        raise TypeError("Need to specify filename and type of source data to store [int(egrand)|(full)term]!")
    if sourcetype in ["int", "term"]:
        sarrname = "source" + sourcetype
        source_logger.debug("Source array type: " + sarrname)
    else:
        raise TypeError("Incorrect source type specified!")
    #Add compression to files and specify good chunkshape
    filters = tables.Filters(complevel=1, complib="zlib") 
    #cshape = (10,10,10) #good mix of t, k, q values
    #Get atom shape for earray
    atomshape = (0, len(k))
    try:
        source_logger.debug("Trying to open source file " + filename)
        rf = tables.openFile(filename, "a", "Source term result")
        if not "results" in rf.root:
            source_logger.debug("Creating group 'results' in source file.")
            resgrp = rf.createGroup(rf.root, "results", "Results")
        else:
            resgrp = rf.root.results
        if not sarrname in resgrp:
            source_logger.debug("Creating array '" + sarrname + "' in source file.")
            sarr = rf.createEArray(resgrp, sarrname, tables.ComplexAtom(itemsize=16), atomshape, filters=filters)
            karr = rf.createEArray(resgrp, "k", tables.Float64Atom(), (0,), filters=filters)
            karr.append(k)
        else:
            source_logger.debug("Source file and node exist. Testing source node shape...")
            sarr = rf.getNode(resgrp, sarrname)
            if sarr.shape[1:] != atomshape[1:]:
                raise ValueError("Source node on file is not correct shape!")
    except IOError:
        raise
    return rf, sarr
