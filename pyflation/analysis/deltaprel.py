''' pyflation.analysis.deltaprel - Module to calculate relative pressure
perturbations.

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.

'''
from __future__ import division
import numpy as np

def soundspeeds(Vphi, phidot, H):
    """Sound speeds of the background fields
    
    Arguments
    ---------
    Vphi: array_like
          First derivative of the potential with respect to the fields
          
    phidot: array_like
            First derivative of the field values with respect to efold number N.
            
    H: array_like
       The Hubble parameter
       
    All the arguments should have the same number of dimensions. Vphi and phidot
    should be arrays of the same size, but H should have a dimension of size 1 
    corresponding to the "field" dimension of the other variables.
    """
    try:
        calphasq = 1 + 2*Vphi/(3*H**2*phidot)
    except ValueError:
        raise ValueError("""Arrays need to have the correct shape.
                            Vphi and phidot should have exactly the same shape,
                            and H should have a dimension of size 1 corresponding
                            to the field dimension of the others.""")
    return calphasq

def rhodots(phidot, H):
    """Derivative in e-fold time of the energy densities of the individual fields.
    
    Arguments
    ---------
    phidot: array_like
            First derivative of the field values with respect to efold number N.
            
    H: array_like
       The Hubble parameter
       
    Both arrays should have the same number of dimensions, but H should have a 
    dimension of size 1 corresponding to the field dimension of phidot.
    """
    return -3*H**3*(phidot**2)

def fullrhodot(phidot, H, axis=-1):
    """Combined derivative in e-fold time of the energy density of the field.
    
    Arguments
    ---------
    phidot: array_like
            First derivative of the field values with respect to efold number N.
            
    H: array_like
       The Hubble parameter
    
    axis: integer, optional
          Specifies which axis is the field dimension, default is the last one.
            
    
    """
    return np.sum(rhodots(phidot, H), axis=axis)

def deltarhosmatrix(Vphi, phidot, H, modes, modesdot, axis):
    """Matrix of the first order perturbed energy densities of the field components.
    
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
    
    #Change shape of phidot, Vphi, H to add extra dimension of modes
    Vphi = np.expand_dims(Vphi, axis+1)
    phidot = np.expand_dims(phidot, axis+1)
    H = np.expand_dims(H, axis+1)
    
    #Do first sum over beta index
    internalsum = np.sum(phidot*modes, axis=axis)
    #Add another dimension to internalsum result
    internalsum = np.expand_dims(internalsum, axis)
    
    result = H**2*phidot*modesdot
    result -= 0.5*H**3*phidot**2*internalsum
    result += Vphi*modes
    
    return result
    
    
    
    

def deltaprelmodes(Vphi, phidot, H, modes, modesdot, axis):
    """Perturbed relative pressure of the fields given as quantum mode functions.
    
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
    
    cs = soundspeeds(Vphi, phidot, H)
    rdots = rhodots(phidot, H)
    rhodot = fullrhodot(phidot, H, axis)
    drhos = deltarhosmatrix(Vphi, phidot, H, modes, modesdot, axis)
    
    res_shape = list(drhos.shape)
    del res_shape[axis]
    
    result = np.zeros(res_shape)
                    
    for ix in np.ndindex(tuple(res_shape[:axis])):
        for i in range(res_shape[axis]):
            for a in range(rdots.shape[axis]):
                for b in range(rdots.shape[axis]):
                    if a != b:
                        result[ix+(i,)] += (1/(2*rhodot[ix]) * (cs[ix+(a,)] - cs[ix+(b,)]) 
                                          * (rdots[ix+(b,)]*drhos[ix+(a,i)] - rdots[ix+(a,)]*drhos[ix+(b,i)]))  
        
    
    return result
    

def deltaprelspectrum(Vphi, phidot, H, modes, modesdot, axis):
    """Power spectrum of the full perturbed relative pressure."""
    dPrelI = deltaprelmodes(Vphi, phidot, H, modes, modesdot, axis)
    
    spectrum = np.sum(dPrelI*dPrelI.conj(), axis=axis)
    
    return spectrum