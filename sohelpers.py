"""Second order helper functions by Ian Huston
    $Id: sohelpers.py,v 1.12 2009/07/02 10:00:08 ith Exp $
    
    Provides helper functions for second order data from cosmomodels.py"""
    
import tables
import numpy as np
import cosmomodels as c
import logging
import time
import os.path
from sourceterm.srccython import getthetaterms
from scipy.integrate import romb

#Start logging
root_log_name = logging.getLogger().name
_log = logging.getLogger(root_log_name + "." + __name__)


def combine_source_and_fofile(sourcefile, fofile, newfile=None):
    """Copy source term to first order file in preparation for second order run.
    
    Parameters
    ----------
    sourcefile: String
                Full path and filename of source file.
                
    fofile: String
            Full path and filename of first order results file.
            
    newfile: String, optional
             Full path and filename of combined file to be created.
             Default is to place the new file in the same directory as the source
             file with the filename having "src" replaced by "foandsrc".
             
    Returns
    -------
    newfile: String
             Full path and filename of saved combined results file.
    """
    if not newfile or not os.path.isdir(os.path.dirname(newfile)):
        newfile = os.path.dirname(fofile) + os.sep + os.path.basename(sourcefile).replace("src", "foandsrc")
        _log.info("Saving combined source and first order results in file %s.", newfile)
    if os.path.isfile(newfile):
        newfile = newfile.replace("foandsrc", "foandsrc_")
    try:
        sf = tables.openFile(sourcefile, "r")
        ff = tables.openFile(fofile, "r")
        nf = tables.openFile(newfile, "w") #Write to new file, not append to old.
    except IOError:
        _log.exception("Source or first order files not found!")
        raise
    
    try:
        try:
            sterm = sf.root.results.sourceterm
            srck = sf.root.results.k
            srcnix = sf.root.results.nix
        except tables.NoSuchNodeError:
            _log.exception("Source term file not in correct format!")
            raise
        fres = ff.root.results
        nres = nf.copyNode(ff.root.results, nf.root)
        #Check that all time steps are calculated
        if len(fres.tresult) != len(srcnix):
            raise ValueError("Not all timesteps have had source term calculated!")
        #Copy first order results
        numks = len(srck)
        yres = fres.yresult.copy(nres, stop=numks)
        tres = fres.tresult.copy(nres)
        tstart = fres.fotstart.copy(nres, stop=numks)
        tstartindex = fres.fotstartindex.copy(nres, stop=numks)
        ystart = fres.foystart.copy(nres, stop=numks)
        params = fres.parameters.copy(nres)
        pot_params = fres.pot_params.copy(nres)
        bgres = nf.copyNode(ff.root.bgresults, nf.root, recursive=True)
        #Copy source term
        nf.copyNode(sterm, nres)
        #Copy source k range
        nf.copyNode(srck, nres)
        _log.info("Source term successfully copied to new file %s.", newfile)
    finally:
        sf.close()
        ff.close()
        nf.close()
    return newfile
        

   
def soderivs_magnitude(m, **kwargs):
    """Equation of motion for second order perturbations including source term"""
    
    #Pick k from kwargs
    kix = kwargs["kix"]
    k = m.k[kix]
    
    if kix is None:
        raise ModelError("Need to specify kix in order to calculate 2nd order perturbation!")
    #Need t index to use first order data
    if kwargs["tix"] is None:
        raise ModelError("Need to specify tix in order to calculate 2nd order perturbation!")
    else:
        tix = kwargs["tix"]
    t = m.tresult[tix]

    #Get first order results for this time step
    fovars = m.yresult[tix,0:7].copy()[:,kix]
    y = m.yresult[tix, 7:].copy()[:,kix]
    phi, phidot, H = fovars[0:3]
    epsilon = m.bgepsilon[tix]
    #get potential from function
    U, dU, d2U, d3U = m.potentials(fovars, m.pot_params)[0:4]        
    
    #Set derivatives taking care of k type
    if type(k) is np.ndarray or type(k) is list: 
        dydx = np.zeros((4,len(k)))
    else:
        dydx = np.zeros(4)
        
    #Get a
    a = m.ainit*np.exp(t)
    #Real parts
    #first term -(3 - epsilon)dp2dot
    dydx[0] = -(3 - epsilon)*y[1]
    
    #second term dp2
    dydx[1] = (- ((k/(a*H))**2)*y[0]
                -(d2U/H**2 - 3*(phidot**2))*y[0])
            
    #Complex 1st term dp2dot
    dydx[2] = -(3 - epsilon)*y[3]
    
    #Complex 2nd term dp2
    dydx[3] = (- ((k/(a*H))**2)*y[2]
                -(d2U/H**2 - 3*(phidot**2))*y[2])
    
    return dydx

def find_soderiv_terms(m, kix=np.array([0])):
    """Run through all time steps finding second order derivative terms."""
    res = []
    for tix, t in enumerate(m.tresult):
        terms = soderivs_magnitude(m, tix=tix, kix=kix)
        res.append(terms)
    return np.array(res)
    
def get_selfconvolution(m, nix, nthetas=257, numsoks=1025):
    """Return self convolution of given function."""
    k = q = m.k[:numsoks]
    theta = np.linspace(0, np.pi, nthetas)
    ie = k, q, theta
    dp1 = m.yresult[nix,3] + m.yresult[nix,5]*1j
    dp1dot = m.yresult[nix,4] + m.yresult[nix,6]*1j
    aterm = getthetaterms(ie, dp1, dp1dot)[0]
    at2 = aterm[0] + aterm[1]*1j
    qm = q[np.newaxis, ...]
    dp1m = dp1[np.newaxis,:numsoks]
    conv = 2*np.pi * qm**2 * dp1m * at2
    integrated = romb(conv, k[1]-k[0])
    return integrated  
        