"""sosource.py Second order source term calculation module.
Author: Ian Huston
$Id: sosource.py,v 1.75 2010/01/18 14:36:02 ith Exp $

Provides the method getsourceandintegrate which uses an instance of a first
order class from cosmomodels to calculate the source term required for second
order models.

"""

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0

import numpy as N
import logging
import time
import os

from romberg import romb
from sourceterm import srccython, srcequations
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

def calculatesource(m, nix, integrand_elements, srceqns):
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
         
    srceqns: Instance of a subclass of srcequations.SourceEquations
             This class should implement the sourceterm function which
             constructs the integrated source term.
         
    Returns
    -------
    s: array_like
       Integrated source term calculated using srcfunc.
    """
    k, q, theta = integrand_elements
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

    #Get integrated source term
    src = srceqns.sourceterm(bgvars, a, potentials, integrand_elements, dphi1, dphi1dot)
        
    if _debug:
        source_logger.debug("Integration successful!")
    return src

           
def getsourceandintegrate(m, savefile=None, srcclass=None, ninit=0, nfinal=-1, ntheta=513, numks=1025):
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
    
    srcclass: class, optional
             Class which contains functions for constructing source term. Defaults to srcequations.SlowRollSource.
             Class should be a subclass of srcequations.SourceEquations and should implement the sourceterm function
             
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
   
    #Check source class and create instance
    if srcclass is None:
        srcclass = srcequations.SlowRollSource
        
    if not issubclass(srcclass, srcequations.SourceEquations):
        raise TypeError("Source class should be a class derived from srcequations.SourceEquations.")
    
    kfx = {"kmin":k[0], "deltak":k[1]-k[0], "numsoks":numks, 
                   "fullkmax":m.k[-1], "nthetas":ntheta}
        
    srceqns = srcclass(kfx)
   
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
                    src = calculatesource(m, nix, integrand_elements, srceqns)
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
    filters = tables.Filters(complevel=1, complib="blosc") 
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
