"""Second order helper functions to set up source term
    $Id: sosource.py,v 1.11 2008/12/01 17:01:58 ith Exp $
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

#Start logging
source_logger = logging.getLogger(__name__)
  
def getsourceintegrand(m, savefile=None):
    """Return source term (slow-roll for now), once first order system has been executed."""
        
    #Initialize variables to store result
    lenmk = len(m.k)
    s2shape = (lenmk, lenmk)
    source_logger.debug("Shape of m.k is " + str(lenmk))
    #Get atom shape for savefile
    atomshape = (0, lenmk, lenmk)
    
    #Set up file for results
    if not savefile or not os.path.isdir(os.path.dirname(savefile)):
        date = time.strftime("%Y%m%d")
        savefile = RESULTSDIR + "source" + date + ".hf5"
        source_logger.info("Saving source results in file " + savefile)

    #Main try block for file IO
    try:
        sf, sarr = opensourcefile(savefile, atomshape, sourcetype="int")
        try:    
            #Main loop over each time step
            for nix, n in enumerate(m.tresult):    
                #Get first order ICs:
                nanfiller = m.getfoystart(m.tresult[nix], N.array([nix]))
                
                #switch nans for ICs in m.yresult
                myr = m.yresult[nix].copy()
                are_nan = N.isnan(myr)
                myr[are_nan] = nanfiller[are_nan]
                
                #Get first order results (from file or variables)
                phi, phidot, H, dphi1real, dphi1dotreal, dphi1imag, dphi1dotimag = [myr[i,:] for i in range(7)]
                dphi1 = dphi1real + dphi1imag*1j
                dphi1dot = dphi1dotreal + dphi1dotimag*1j
                pottuple = m.potentials(myr)
                #Get potentials in right shape
                pt = []
                for p in pottuple:
                    if N.shape(p) != N.shape(pottuple[0]):
                        pt.append(p*N.ones_like(pottuple[0]))
                    else:
                        pt.append(p)
                U, dU, dU2, dU3 = pt
                
                #Single time step
                a = m.ainit*N.exp(n)
                
                #Initialize result variable for k modes
                s2 = N.empty(s2shape)
                for kix, k in enumerate(m.k):
                    #Single k mode
                    #Result variable for source
                    s1 = N.empty_like(m.k)
                    for qix, q in enumerate(m.k):
                        #Single q mode
                        #Check abs(qix-kix)-1 is not negative
                        dphi1ix = N.abs(qix-kix) -1
                        if dphi1ix < 0:
                            dp1diff = dp1dotdiff = 0
                        else:
                            dp1diff = dphi1[dphi1ix]
                            dp1dotdiff = dphi1dot[dphi1ix]
                        
                        #First major term:
                        term1 = (1/(2*N.pi**2) * (1/H[kix]**2) * (dU3[kix] + 3*phidot[kix]*dU2[kix]) 
                                    * q**2*dp1diff*dphi1[qix])
                        #Second major term:
                        term2 = (1/(2*N.pi**2) * ((1/(a*H[kix]) + 0.5)*q**2 - 2*(q**4/k**2)) * dp1dotdiff * dphi1dot[qix])
                        #Third major term:
                        term3 = (1/(2*N.pi**2) * 1/(a*H[kix])**2 * (2*(q**6/k**2) + 2.5*q**4 + 2*(k*q)**2) * phidot[kix] 
                                    * dp1diff * dphi1[qix])
                        s1[qix] = term1 + term2 + term3
                    #add sourceterm for each q
                    s2[kix] = s1
                #save results for each q
                sarr.append(s2[N.newaxis])
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
    try:
        source_logger.debug("Trying to open source file " + filename)
        rf = tables.openFile(filename, "a", "Source term result")
        if not "results" in rf.root:
            source_logger.debug("Creating group 'results' in source file.")
            rf.createGroup(rf.root, "results", "Results")
        if not sarrname in rf.root.results:
            source_logger.debug("Creating array '" + sarrname + "' in source file.")
            sarr = rf.createEArray(rf.root.results, sarrname, tables.Float64Atom(), atomshape)
        else: 
            sarr = rf.getNode(rf.root.results, sarrname)
            if sarr.shape[1:] != atomshape[1:]:
                raise ValueError("EArray on file is not correct shape!")
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
