''' pyflation.analysis.utilities - Helper module for analysis package

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.

'''
from __future__ import division
import numpy as np

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
    ix2out = slist.pop(ix2) #@UnusedVariable
    slist[ix1] = nfields**2
    return modematrix.reshape(slist) 

def components_from_model(m, tix=None, kix=None):
    """Get the component variables of delta Prel from a model instance.
    
    Arguments
    ---------
    m: Cosmomodels model instance
       The model instance with which to perform the calculation
       
    tix: integer
         Index for timestep at which to perform calculation. Default is to 
         calculate over all timesteps.
        
    kix: integer
         Index for k mode for which to perform the calculation. Default is to
         calculate over all k modes.
         
    Returns
    -------
    Tuple containing:
    
    Vphi: array_like
          First derivative of the potential with respect to the fields
          
    phidot: array_like
            First derivative of the field values with respect to efold number N.
            
    H: array_like
       The Hubble parameter
       
    modes: array_like
           Mode matrix of first order perturbations. Component array should
           have two dimensions of length nfields.
    
    modesdot: array_like
           Mode matrix of N-derivative of first order perturbations. Component array should
           have two dimensions of length nfields.
    
    axis: integer
          Specifies which axis is first in mode matrix, e.g. if modes has shape
          (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
          assumed to be beside each other so (100,3,10,3) would not be valid.
    """
    
        
    if tix is None:
        tslice = slice(None)
    else:
        #Check for negative tix
        if tix < 0:
            tix = len(m.tresult) + tix
        tslice = slice(tix, tix+1)
    if kix is None:
        kslice = slice(None)
    else:
        kslice = slice(kix, kix+1)
        
    
    
    phidot = m.yresult[tslice, m.phidots_ix, kslice]
    H = m.yresult[tslice, m.H_ix, kslice]
    Vphi = np.array([m.potentials(myr, m.pot_params)[1] 
                     for myr in m.yresult[tslice,m.bg_ix,kslice]])
    modes = getmodematrix(m.yresult[tslice, m.dps_ix, kslice], m.nfields)
    modesdot = getmodematrix(m.yresult[tslice, m.dpdots_ix, kslice], m.nfields)
    axis = 1
    Vphi, phidot, H, modes, modesdot, axis = correct_shapes(Vphi, phidot, H, modes, modesdot, axis)
    
    return Vphi,phidot,H,modes,modesdot,axis

def makespectrum(modes_I, axis):
    """Calculate spectrum of input.
    
    Spectrum = \Sum_I modes_I * modes_I.conjugate, 
    where sum over I is on the dimension denoted by axis
    
    Arguments
    ---------
    modes_I: array_like
             Array of (complex) values to calculate spectrum with.
    
    axis: integer
          Dimension along which to sum the spectrum over.
          
    Returns
    -------
    spectrum: array_like, float
              Real valued array of calculated spectrum.
    """
    modes_I = np.atleast_1d(modes_I)
    spectrum = np.sum(modes_I * modes_I.conj(), axis=axis).astype(np.float)
    return spectrum

def correct_shapes(Vphi, phidot, H, modes, modesdot, axis):
    """Return variables with the correct shapes for calculation.
    
    Arguments
    ---------
    Vphi: array_like
          First derivative of the potential with respect to the fields
          
    phidot: array_like
            First derivative of the field values with respect to efold number N.
            
    H: array_like
       The Hubble parameter
       
    modes: array_like
           Mode matrix of first order perturbations. Component array should
           have two dimensions of length nfields.
           
    modesdot: array_like
           Mode matrix of N-derivative of first order perturbations. Component array should
           have two dimensions of length nfields.
           
    axis: integer
          Specifies which axis is first in mode matrix, e.g. if modes has shape
          (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
          assumed to be beside each other so (100,3,10,3) would not be valid.
          
    Returns
    -------
    
    result: tuple
            Tuple of the variables with correct shapes in the form
            (Vphi, phidot, H, modes, modesdot, axis)
           
    """
    mshape = modes.shape
    if mshape[axis+1] != mshape[axis]:
        raise ValueError("The mode matrix dimensions are not together.")
    if mshape != modesdot.shape:
        raise ValueError("Mode matrix and its derivative should be the same shape.")
    mshapelist = list(mshape)
    del mshapelist[axis]
    
    #Make Vphi, phidot and H into at least 1-d arrays
    Vphi, phidot, H = np.atleast_1d(Vphi, phidot, H)
    
    #If Vphi doesn't have k axis then add it
    if len(Vphi.shape) < len(phidot.shape):
        Vphi = np.expand_dims(Vphi, axis=-1) 
    
    if len(mshapelist) != len(Vphi.shape) != len(phidot.shape):
        raise ValueError("Vphi, phidot and modes arrays must have correct shape.")
    
    #If H doesn't have a field axis then add one
    if len(H.shape) < len(phidot.shape):
        H = np.expand_dims(H, axis)
    
    return Vphi, phidot, H, modes, modesdot, axis
