"""Second order helper functions to set up source term
    $Id: sosource.py,v 1.22 2008/12/02 17:57:28 ith Exp $
    """

from __future__ import division # Get rid of integer division problems, i.e. 1/2=0
import tables
import numpy as N
from scipy import integrate
import helpers
import logging
import time
import os

RESULTSDIR = "/misc/scratch/ith/numerics/results/"
# RESULTSDIR = "/starpc34/scratch/ith/numerics/results/"

#Start logging
source_logger = logging.getLogger(__name__)
  
def getsourceintegrand(m, savefile=None):
    """Return source term (slow-roll for now), once first order system has been executed."""
    #Debugging
    nixend = 10
    
    #Initialize variables to store result
    lenmk = len(m.k)
    s2shape = (lenmk, lenmk)
    source_logger.debug("Shape of m.k is %s.", str(lenmk))
    #Get atom shape for savefile
    atomshape = (0, lenmk, lenmk)
    
    #Set up file for results
    if not savefile or not os.path.isdir(os.path.dirname(savefile)):
        date = time.strftime("%Y%m%d%H%M%S")
        savefile = RESULTSDIR + "source" + date + ".hf5"
        source_logger.info("Saving source results in file " + savefile)

    #Main try block for file IO
    try:
        sf, sarr = opensourcefile(savefile, atomshape, sourcetype="int")
        try:
            source_logger.debug("Entering main time loop...")    
            #Main loop over each time step
            for nix, n in enumerate(m.tresult[:nixend]):    
                if N.ceil(n) == n:
                    source_logger.debug("Starting n=" + str(n) + " sequence...")
                #Get first order ICs:
                nanfiller = m.getfoystart(m.tresult[nix].copy(), N.array([nix]))
                source_logger.debug("Left getfoystart. Filling nans...")
                #switch nans for ICs in m.yresult
                myr = m.yresult[nix].copy()
                are_nan = N.isnan(myr)
                myr[are_nan] = nanfiller[are_nan]
                source_logger.debug("NaNs filled. Setting dynamical variables...")
                
                #Get first order results (from file or variables)
                phi, phidot, H, dphi1real, dphi1dotreal, dphi1imag, dphi1dotimag = [myr[i,:] for i in range(7)]
                dphi1 = dphi1real + dphi1imag*1j
                dphi1dot = dphi1dotreal + dphi1dotimag*1j
                source_logger.debug("Variables set. Getting potentials for this timestep...")
                pottuple = m.potentials(myr)
                #Get potentials in right shape
                pt = []
                for p in pottuple:
                    if N.shape(p) != N.shape(pottuple[0]):
                        pt.append(p*N.ones_like(pottuple[0]))
                    else:
                        pt.append(p)
                U, dU, dU2, dU3 = pt
                source_logger.debug("Potentials obtained. Setting a and making results array...")
                #Single time step
                a = m.ainit*N.exp(n)
                
                #Initialize result variable for k modes
                s2 = N.empty(s2shape)
                
                source_logger.debug("Starting main k loop...")
                #Get k indices
                kix = N.arange(lenmk)
                qix = N.arange(lenmk)
                #Check abs(qix-kix)-1 is not negative
                dphi1ix = N.abs(qix[:, N.newaxis]-kix) -1
                dp1diff = N.where(dphi1ix < 0, 0, dphi1[dphi1ix])
                dp1dotdiff = N.where(dphi1ix <0, 0, dphi1dot[dphi1ix])                    
                #temp k and q vars
                k = m.k[kix]
                q = m.k[qix]
                #First major term:
                s2 = (1/(2*N.pi**2) * (1/H**2) * (dU3 + 3*phidot*dU2) 
                            * q**2*dp1diff*dphi1)
                #Second major term:
                s2 += (1/(2*N.pi**2) * ((1/(a*H) + 0.5)*q**2 - 2*(q**4/k**2)) * dp1dotdiff * dphi1dot)
                #Third major term:
                s2 += (1/(2*N.pi**2) * 1/(a*H)**2 * (2*(q**6/k**2) + 2.5*q**4 + 2*(k*q)**2) * phidot
                                            * dp1diff * dphi1)
#                 s2 = term1 + term2 + term3
                #save results for each q
                source_logger.debug("Saving results for this tstep...")
                sarr.append(s2[N.newaxis])
                source_logger.debug("Results for this tstep saved.")
        finally:
            #source = N.array(source)
            sf.close()
    except IOError:
        raise
    return savefile
            
def getsource(m, intmethod=None, fullinfo=False):
    """Return integrated source function for model m using romberg integration."""
    #Choose integration method
    if intmethod is None:
        try:
            if all(m.k[1:]-m.k[:-1] == m.k[1]-m.k[0]) and helpers.ispower2(len(m.k)-1):
                intmethod = "romb"
            else:
                intmethod = "simps"
        except IndexError:
                raise IndexError("Need more than one k to calculate integral!")
    #Now proceed with integration
    if intmethod is "romb":
        if not helpers.ispower2(len(m.k)-1):
            raise AttributeError("Need to have 2**n + 1 different k values for integration.")
        msource = integrate.romb(getsourceintegrand(m))
    elif intmethod is "simps":
        msource = integrate.simps(getsourceintegrand(m), m.k)
    else:
        raise ValueError("Need to specify correct integration method!")
    #Check if we want data about integration
    if fullinfo:
        results = [msource, intmethod]
    else:
        results = msource
    return results

def opensourcefile(filename, atomshape, sourcetype=None):
    """Open the source term hdf5 file with filename."""
    if not filename or not sourcetype:
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
            sarr = rf.createEArray(rf.root.results, sarrname, tables.Float64Atom(), atomshape, filters=filters)
        else:
            source_logger.debug("Source file and node exist. Testing source node shape...")
            sarr = rf.getNode(rf.root.results, sarrname)
            if sarr.shape[1:] != atomshape[1:]:
                raise ValueError("Source node on file is not correct shape!")
    except IOError:
        raise
    return rf, sarr
    
def savetofile(filename, m, sourceterm=None):
    """Save the source term to the hdf5 file with filename."""
    if not filename or not m:
        raise TypeError("Need to specify both filename and model variable.")
    if sourceterm is None:
        print "No source term given, calculating now..."
        sourceterm = getsource(m)
                
    try:
        rf = tables.openFile(filename, "a", "Source term result")
        try:
            if not "results" in rf.root:
                rf.createGroup(rf.root, "results", "Results")
            if not "sourceterm" in rf.root.results:
                stab = rf.createTable(rf.root.results, "sourceterm", sourcetermdict(m, sourceterm))
            else: 
                stab = rf.root.results.sourceterm

            stab.flush()
        finally:
            rf.close()
    except IOError:
        raise

def sourcetermdict(m, sourceterm):
    """Return dictionary with source term table configuration for HDF5 file"""
    if not m or sourceterm is None:
        raise TypeError("Need to specify both model and sourceterm.")
    sdict = {
    "k" : tables.Float64Col(),
    "source" : tables.Float64Col(sourceterm[:,0].shape)}
    
    return sdict
