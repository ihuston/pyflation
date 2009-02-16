"""sosource.py Second order source term calculation module.
Author: Ian Huston
$Id: sosource.py,v 1.41 2009/02/16 17:32:22 ith Exp $

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

#This is the results directory which will be used if no filenames are specified
RESULTSDIR = "/misc/scratch/ith/numerics/results/"

#Start logging
source_logger = logging.getLogger(__name__)

def slowrollsrcterm(k, q, a, potentials, bgvars, fovars, s2shape):
    """Return unintegrated slow roll source term.
    
    The source term before integration is calculated here using the slow roll
    approximation. This function follows the revised version of Eq (5.8) in 
    Malik 06 (astro-ph/0610864v5).
    
    Parameters
    ----------
    k: array_like
       Array of mode numbers being calculated.
    
    q: array_like
       Array of mode numbers which will be integrated over. 
       (Should be the same as k, might remove later.)
       
    a: float
       Scale factor at the current timestep, `a = ainit*exp(n)`
    
    potentials: tuple
                Tuple of potential values in the form `(U, dU, dU2, dU3)`
       
    bgvars: tuple
            Tuple of background field values in the form `(phi, phidot, H)`
            
    fovars: tuple
            Tuple of first order field values in the form `(dphi1, dphi1dot, dp1diff, dp1dotdiff)`
            
    s2shape: tuple
             Shape of the results array to return
             
    Returns
    -------
    s2: array_like
        Array containing the unintegrated source terms for all k and q modes.
        
    References
    ----------
    Malik, K. 2006, JCAP03(2007)004, astro-ph/0610864v5
    """
    #Unpack variables
    phi, phidot, H = bgvars
    U, dU, dU2, dU3 = potentials
    dphi1, dphi1dot, dp1diff, dp1dotdiff = fovars
    #Initialize result variable for k modes
    s2 = N.empty(s2shape)
                
    #Calculate unintegrated source term
    #First major term:
    s2 = ((1/H**2) * (dU3 + 3*phidot*dU2) * q**2*dp1diff*dphi1)
    #Second major term:
    s2 += ((-1.5*q**2) * phidot * dp1dotdiff * dphi1dot)
    #Third major term:
    s2 += (1/((a*H)**2) * (2.5*q**4 + 2*(k*q)**2) * phidot * dp1diff * dphi1)
    #Multiply by prefactor
    s2 = s2 * (1/(2*N.pi**2))
    
    return s2

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
         integrand_elements = (k, q, theta, klessq)
         
    srcfunc: function, optional
             Funtion which contains expression for source integrand
             Default is slowrollsrcterm in this module.        
         
    Returns
    -------
    s: array_like
       Integrated source term calculated using srcfunc.
    """
        #testing
#     nixend = 10
    k, q, theta, klessq = integrand_elements
    #Initialize variables to store result
    lenmk = len(m.k)
    s2shape = (lenmk, lenmk)
    source_logger.debug("Shape of m.k is %s.", str(lenmk))
    
    #Copy of yresult
    myr = m.yresult[nix].copy()
    if N.any(n < m.fotstart):
        #Get first order ICs:
        nanfiller = m.getfoystart(m.tresult[nix].copy(), N.array([nix]))
        source_logger.debug("Left getfoystart. Filling nans...")
        #switch nans for ICs in m.yresult
        are_nan = N.isnan(myr)
        myr[are_nan] = nanfiller[are_nan]
        source_logger.debug("NaNs filled. Setting dynamical variables...")
                
    #Get first order results (from file or variables)
    bgvars = myr[0:3,:]
    dphi1 = myr[3,:] + myr[4,:]*1j
    dphi1dot = myr[5,:] + myr[6,:]*1j
    source_logger.debug("Variables set. Getting potentials for this timestep...")
    pottemp = m.potentials(myr)
    #Get potentials in right shape
    potentials = []
    for p in pottemp:
        if N.shape(p) != N.shape(pottemp[0]):
            potentials.append(p*N.ones_like(pottemp[0]))
        else:
            potentials.append(p)
    source_logger.debug("Potentials obtained. Setting a and making results array...")
    #Single time step
    a = m.ainit*N.exp(n)
    #Get k indices
    kix = N.arange(lenmk)
    qix = N.arange(lenmk)
    #Check abs(qix-kix)-1 is not negative and get q-k variables
    dphi1ix = N.abs(kix-qix[:, N.newaxis]) -1
    dp1diff = N.where(dphi1ix < 0, 0, dphi1[dphi1ix])
    dp1dotdiff = N.where(dphi1ix <0, 0, dphi1dot[dphi1ix])
    
    fovars = (dphi1, dphi1dot, dp1diff, dp1dotdiff) #First order variables for src function                 
    #temp k and q vars
    k = m.k[kix]
    q = m.k[qix]
    
                
    #save results for each q
    source_logger.debug("Integrating source term for this tstep...")
    #Get unintegrated source term
    s2 = srcfunc(k, q, a, potentials, bgvars, fovars, s2shape, intfunc)
    
    #Get integration function
    intfunc, intfnargs = helpers.getintfunc(m.k)
    #Do integration
    s2int=intfunc(s2, **intfnargs)
    return s2int
                
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
             Function signature is `srcfunc(k, q, a, potentials, bgvars, fovars, s2shape)`.
               
    Returns
    -------
    savefile: String
              Filename where results have been saved.
    """
    #Initialize variables for all timesteps
    halfk = m.k[:len(m.k)/2] #Need N=len(m.k)/2 for case when q=-k
    k = halfk[...,newaxis,newaxis]
    q = halfk[newaxis,...,newaxis]
    theta = N.linspace(0, N.pi, ntheta)[newaxis,newaxis,...]
    #Main array of k^i - q^i values
    klessq=sqrt(k**2+q**2-2*k*q*cos(theta2))
    #Pack together in tuple
    integrand_elements = (k, q, theta, klessq)
    #Get atom shape for savefile
    atomshape = (0, len(halfk))
    #Main try block for file IO
    try:
        sf, sarr = opensourcefile(atomshape, savefile, sourcetype="term")
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

def opensourcefile(atomshape, filename=None, sourcetype=None):
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
    try:
        source_logger.debug("Trying to open source file " + filename)
        rf = tables.openFile(filename, "a", "Source term result")
        if not "results" in rf.root:
            source_logger.debug("Creating group 'results' in source file.")
            rf.createGroup(rf.root, "results", "Results")
        if not sarrname in rf.root.results:
            source_logger.debug("Creating array '" + sarrname + "' in source file.")
            sarr = rf.createEArray(rf.root.results, sarrname, tables.ComplexAtom(itemsize=16), atomshape, filters=filters)
        else:
            source_logger.debug("Source file and node exist. Testing source node shape...")
            sarr = rf.getNode(rf.root.results, sarrname)
            if sarr.shape[1:] != atomshape[1:]:
                raise ValueError("Source node on file is not correct shape!")
    except IOError:
        raise
    return rf, sarr
