''' pyflation.analysis.spectrum - Explanation

Author: ith
'''

import numpy as np
from scipy import interpolate

def getmodematrix(y, nfields, ix=None, ixslice=None):
    """Helper function to reshape flat nfield^2 long y variable into nfield*nfield mode
    matrix. Returns a view of the y array (changes will be reflected in underlying array).
    
    Parameters
    ----------
    ixslice: index slice, optional
        The index slice of y to use, defaults to full extent of y.
        
    Returns
    -------
    
    result: view of y array with shape nfield*nfield structure
    """
    if ix is None:
        #Use second dimension for index slice by default
        ix = 1
    if ixslice is None:
        #Assume slice is full extent if none given.
        ixslice = slice(None)
    indices = [Ellipsis]*len(y.shape)
    indices[ix] = ixslice
    modes = y[tuple(indices)]
        
    s = list(modes.shape)
    #Check resulting array is correct shape
    if s[ix] != nfields**2:
        raise ValueError("Array does not have correct dimensions of nfields**2.")
    s[ix] = nfields
    s.insert(ix+1, nfields)
    result = modes.reshape(s)
    return result

def flattenmodematrix(modematrix, nfields, ix1=None, ix2=None):
    """Flatten the mode matrix given into nfield^2 long vector."""
    s = modematrix.shape
    if s.count(nfields) < 2:
        raise ValueError("Mode matrix does not have two nfield long dimensions.")
    try:
        #If indices are not specified, use first two in order
        if ix1 is None:
            ix1 = s.index(nfields)
        if ix2 is None:
            #The second index is assumed to be after ix1
            ix2 = s.index(nfields, ix1+1)
    except ValueError:
        raise ValueError("Cannot determine correct indices for nfield long dimensions!")
    slist = list(s)
    ix2out = slist.pop(ix2)
    slist[ix1] = nfields**2
    return modematrix.reshape(slist) 
    
    



def Pphi(m, recompute=False):
    """Return the spectrum of scalar perturbations P_phi for each field and k.
    
    This is the unscaled version $P_{\phi}$ which is related to the scaled version by
    $\mathcal{P}_{\phi} = k^3/(2pi^2) P_{\phi}$. Note that result is stored as the
    instance variable m.Pphi.
    For multifield systems the full crossterm matrix is returned which 
    has shape nfields*nfields flattened down to a vector of length nfields^2. 
    
    Parameters
    ----------
    recompute: boolean, optional
               Should value be recomputed even if already stored? Default is False.
    
    Returns
    -------
    Pphi: array_like, dtype: float64
          3-d array of Pphi values for all timesteps, fields and k modes
    """
         
    #Get into mode matrix form, over first axis   
    mdp = getmodematrix(m.yresult, m.nfields, ix=1, ixslice=m.dps_ix)
    #Take tensor product of modes and conjugate, summing over second mode
    #index.
    mPphi = np.zeros_like(mdp)
    #Do for loop as tensordot too memory expensive
    nfields=m.nfields
    for i in range(nfields):
        for j in range(nfields):
            for k in range(nfields):
                mPphi[:,i,j] += mdp[:,i,k]*mdp[:,j,k].conj() 
    #Flatten back into vector form
    Pphi = flattenmodematrix(mPphi, m.nfields, 1, 2) 
    return Pphi


def findns(m, k=None, nix=-1):
    """Return the value of n_s
    
    Arguments
    ---------
    m: Cosmomodels instance
       Model for which to calculate n_s
       
    k: array_like, optional
       k values at which to calculate n_s (should be inside range of ks used in model)
       Defaults to all k values in m.
       
    nix: integer, optional
         Timestep at which to calculate n_s, defaults to last timestep
         
    Returns
    -------
    n_s: float
         The value of the spectral index at the requested k value and timestep
    """
    
    #If k is not defined, get value at all m.k
    if k is None:
        k = m.k
    else:
        if np.any(k<m.k.min() or k>m.k.max()):
            m._log.warn("Warning: Extrapolating to k value outside those used in spline!")
    
    xp = np.log(Pr(m)[nix])
    lnk = np.log(k)
    
    #Need to sort into ascending k
    sortix = lnk.argsort()
            
    #Use cubic splines to find deriv
    tck = interpolate.splrep(lnk[sortix], xp[sortix])
    ders = interpolate.splev(lnk[sortix], tck, der=1)
    
    ns = 1 + ders
    #Unorder the ks again
    nsunsort = np.zeros(len(ns))
    nsunsort[sortix] = ns
    
    return nsunsort

def findHorizoncrossings(m, factor=1):
    """Find horizon crossing for all ks"""
    return m.findallkcrossings(m.tresult, m.yresult[:,2], factor)
     
     
def Pr(m):
    """Return the spectrum of curvature perturbations $P_R$ for each k.
    
    For a multifield model this is given by:
    
    Pr = (\Sum_K \dot{\phi_K}^2 )^{-2} 
            \Sum_{I,J} \dot{\phi_I} \dot{\phi_J} P_{IJ}
            
    where P_{IJ} = \Sum_K \chi_{IK} \chi_JK}
    and \chi are the mode matrix elements.  
    
    This is the unscaled version $P_R$ which is related to the scaled version by
    $\mathcal{P}_R = k^3/(2pi^2) P_R$. Note that result is stored as the instance variable
    m.Pr. 
    
    Parameters
    ----------
    recompute: boolean, optional
               Should value be recomputed even if already stored? Default is False.
               
    Returns
    -------
    Pr: array_like, dtype: float64
        Array of Pr values for all timesteps and k modes
    """      
    phidot = np.float64(m.yresult[:,m.phidots_ix,:]) #bg phidot
    phidotsumsq = (np.sum(phidot**2, axis=1))**2
    #Get mode matrix for Pphi as nfield*nfield
    Pphimatrix = getmodematrix(Pphi(m), m.nfields, 1, slice(None))
    #Multiply mode matrix by corresponding phidot value
    summatrix = phidot[:,np.newaxis,:,:]*phidot[:,:,np.newaxis,:]*Pphimatrix
    #Flatten mode matrix and sum over all nfield**2 values
    sumflat = np.sum(flattenmodematrix(summatrix, m.nfields, 1, 2), axis=1)
    #Divide by total sum of derivative terms
    Pr = sumflat/phidotsumsq
    return Pr


def calPr(m):
    """Return the spectrum of curvature perturbations $\mathcal{P}_\mathcal{R}$ 
    for each timestep and k mode.
    
    This is the scaled power spectrum which is related to the unscaled version by
    $\mathcal{P}_\mathcal{R} = k^3/(2pi^2) P_\mathcal{R}$. 
     
    Returns
    -------
    calPr: array_like
        Array of Pr values for all timesteps and k modes
    """
    #Basic caching of result
    return 1/(2*np.pi**2) * m.k**3 * Pr(m)           
