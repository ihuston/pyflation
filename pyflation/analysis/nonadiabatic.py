''' pyflation.analysis.nonadiabatic - Module to calculate relative pressure
perturbations.

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.

'''
from __future__ import division
import numpy as np

import utilities

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

def totalsoundspeed(Vphi, phidot, H, axis):
    """Total sound speed of the fluids
    
    Arguments
    ---------
    Vphi: array_like
          First derivative of the potential with respect to the fields
          
    phidot: array_like
            First derivative of the field values with respect to efold number N.
            
    H: array_like
       The Hubble parameter
       
    axis: integer
          Index of dimension to sum over (field dimension).
       
    All the arguments should have the same number of dimensions. Vphi and phidot
    should be arrays of the same size, but H should have a dimension of size 1 
    corresponding to the "field" dimension of the other variables.
    
    Returns
    -------
    csq: array_like
         The total sound speed of the fluid, csq = P'/rho'
    """
    
    try:
        csq = 1 + 2*np.sum(Vphi*phidot, axis=axis)/(3*np.sum((H*phidot)**2, axis=axis))
    except ValueError:
        raise ValueError("""Arrays need to have the correct shape.
                            Vphi and phidot should have exactly the same shape,
                            and H should have a dimension of size 1 corresponding
                            to the field dimension of the others.""")
    return csq

def Pdots(Vphi, phidot, H):
    """Derivative of pressure of the background fields
    
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
        Pdotalpha = -(2*phidot*Vphi + 3*H**2*phidot**2)
    except ValueError:
        raise ValueError("""Arrays need to have the correct shape.
                            Vphi and phidot should have exactly the same shape,
                            and H should have a dimension of size 1 corresponding
                            to the field dimension of the others.""")
    return Pdotalpha

def fullPdot(Vphi, phidot, H, axis=-1):
    """Combined derivative in e-fold time of the pressure of the fields.
    
    Arguments
    ---------
    Vphi: array_like
          First derivative of the potential with respect to the fields
          
    phidot: array_like
            First derivative of the field values with respect to efold number N.
            
    H: array_like
       The Hubble parameter
    
    axis: integer, optional
          Specifies which axis is the field dimension, default is the last one.
            
    
    """
    return np.sum(Pdots(Vphi, phidot, H), axis=axis)

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
    return -3*H**2*(phidot**2)

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
    
    Returns
    -------
    result: array_like
            The matrix of the first order perturbed energy densities.
    
    """
    Vphi, phidot, H, modes, modesdot, axis = utilities.correct_shapes(Vphi, phidot, 
                                                        H, modes, modesdot, axis)
    
    #Change shape of phidot, Vphi, H to add extra dimension of modes
    Vphi = np.expand_dims(Vphi, axis+1)
    phidot = np.expand_dims(phidot, axis+1)
    H = np.expand_dims(H, axis+1)
    
    #Do first sum over beta index
    internalsum = np.sum(phidot*modes, axis=axis)
    #Add another dimension to internalsum result
    internalsum = np.expand_dims(internalsum, axis)
    
    result = H**2*phidot*modesdot
    result -= 0.5*H**2*phidot**2*internalsum
    result += Vphi*modes
    
    return result

def deltaPmatrix(Vphi, phidot, H, modes, modesdot, axis):
    """Matrix of the first order perturbed pressure of the field components.
    
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
    result: array_like
            The matrix of the first order perturbed pressure.
    
    """
    Vphi, phidot, H, modes, modesdot, axis = utilities.correct_shapes(Vphi, phidot, H, modes, modesdot, axis)
    
    #Change shape of phidot, Vphi, H to add extra dimension of modes
    Vphi = np.expand_dims(Vphi, axis+1)
    phidot = np.expand_dims(phidot, axis+1)
    H = np.expand_dims(H, axis+1)
    
    #Do first sum over beta index
    internalsum = np.sum(phidot*modes, axis=axis)
    #Add another dimension to internalsum result
    internalsum = np.expand_dims(internalsum, axis)
    
    result = H**2*phidot*modesdot
    result -= 0.5*H**2*phidot**2*internalsum
    result -= Vphi*modes
    
    return result

def deltaPrelmodes(Vphi, phidot, H, modes, modesdot, axis):
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
    
    Vphi, phidot, H, modes, modesdot, axis = utilities.correct_shapes(Vphi, phidot, H, modes, modesdot, axis)
    
    cs = soundspeeds(Vphi, phidot, H)
    rdots = rhodots(phidot, H)
    rhodot = fullrhodot(phidot, H, axis)
    drhos = deltarhosmatrix(Vphi, phidot, H, modes, modesdot, axis)
    
    res_shape = list(drhos.shape)
    del res_shape[axis]
    
    result = np.zeros(res_shape, dtype=modes.dtype)
                    
    for ix in np.ndindex(tuple(res_shape[:axis])):
        for i in range(res_shape[axis]):
            for a in range(rdots.shape[axis]):
                for b in range(rdots.shape[axis]):
                    if a != b:
                        result[ix+(i,)] += (1/(2*rhodot[ix]) * (cs[ix+(a,)] - cs[ix+(b,)]) 
                                          * (rdots[ix+(b,)]*drhos[ix+(a,i)] - rdots[ix+(a,)]*drhos[ix+(b,i)]))  
        
    
    return result

def deltaPnadmodes(Vphi, phidot, H, modes, modesdot, axis):
    """Perturbed non-adiabatic pressure of the fields given as quantum mode functions.
    
    
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
    
    Vphi, phidot, H, modes, modesdot, axis = utilities.correct_shapes(Vphi, phidot, H, modes, modesdot, axis)
    
    csq = totalsoundspeed(Vphi, phidot, H, axis)
    csshape = csq.shape
    # Add two dimensions corresponding to mode axes
    csq.resize(csshape[:axis] + (1,1) + csshape[axis:])
    dP = deltaPmatrix(Vphi, phidot, H, modes, modesdot, axis)
    drhos = deltarhosmatrix(Vphi, phidot, H, modes, modesdot, axis)
    
    result = np.sum(dP - csq*drhos, axis=axis)
        
    
    return result    

def Smodes(Vphi, phidot, H, modes, modesdot, axis):
    """Isocurvature perturbation S of the fields given as quantum mode functions.
    
    
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
    
    dpnadmodes = deltaPnadmodes(Vphi, phidot, H, modes, modesdot, axis)
    Pdot = fullPdot(Vphi, phidot, H, axis)
    Pdot = np.expand_dims(Pdot, axis)
    result = Pdot*dpnadmodes       
    
    return result  

def deltarhospectrum(Vphi, phidot, H, modes, modesdot, axis):
    """Power spectrum of the perturbed energy density.
    
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
    deltarhospectrum: array
                      Spectrum of the perturbed energy density
    """
    drhomodes = deltarhosmatrix(Vphi, phidot, H, modes, modesdot, axis)
    
    drhoI = np.sum(drhomodes, axis=axis)
    
    spectrum = utilities.makespectrum(drhoI, axis)
    
    return spectrum

def deltaPspectrum(Vphi, phidot, H, modes, modesdot, axis):
    """Power spectrum of the full perturbed relative pressure.
    
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
    deltaPspectrum: array
                    Spectrum of the perturbed pressure
    """
    dPmodes = deltaPmatrix(Vphi, phidot, H, modes, modesdot, axis)
    
    dPI = np.sum(dPmodes, axis=axis)
    
    spectrum = utilities.makespectrum(dPI, axis)
    
    return spectrum

def deltaPrelspectrum(Vphi, phidot, H, modes, modesdot, axis):
    """Power spectrum of the full perturbed relative pressure.
    
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
    deltaPrelspectrum: array
                      Spectrum of the perturbed relative pressure
    """
    dPrelI = deltaPrelmodes(Vphi, phidot, H, modes, modesdot, axis)
    
    spectrum = utilities.makespectrum(dPrelI, axis)
    
    return spectrum

def deltaPnadspectrum(Vphi, phidot, H, modes, modesdot, axis):
    """Power spectrum of the full perturbed non-adiabatic pressure.
    
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
    deltaPnadspectrum: array
                      Spectrum of the non-adiabatic pressure perturbation
    """
    dPrelI = deltaPnadmodes(Vphi, phidot, H, modes, modesdot, axis)
    
    spectrum = utilities.makespectrum(dPrelI, axis)
    
    return spectrum

def Sspectrum(Vphi, phidot, H, modes, modesdot, axis):
    """Power spectrum of the isocurvature perturbation S.
    
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
    Sspectrum: array
               Spectrum of the isocurvature perturbation S.
    """
    dSI = Smodes(Vphi, phidot, H, modes, modesdot, axis)
    
    spectrum = utilities.makespectrum(dSI, axis)
    
    return spectrum

def scaled_dPnad_spectrum(Vphi, phidot, H, modes, modesdot, axis, k):
    """Power spectrum of delta Pnad scaled with k^3/(2*pi^2)
    
    Assumes that k dimension is last.
    
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
    scaled_dPnad_spectrum: array
                           Scaled spectrum of the non-adiabatic pressure 
                           perturation.
    """
    spectrum = deltaPnadspectrum(Vphi, phidot, H, modes, modesdot, axis)
    #Add extra dimensions to k if necessary
    scaled_spectrum = k**3/(2*np.pi**2) * spectrum
    return scaled_spectrum

def scaled_dP_spectrum(Vphi, phidot, H, modes, modesdot, axis, k):
    """Power spectrum of delta P scaled with k^3/(2*pi^2)
    
    Assumes that k dimension is last.
    
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
    scaled_dP_spectrum: array
                        Scaled spectrum of the perturbed pressure
    """
    spectrum = deltaPspectrum(Vphi, phidot, H, modes, modesdot, axis)
    #Add extra dimensions to k if necessary
    scaled_spectrum = k**3/(2*np.pi**2) * spectrum
    return scaled_spectrum

def scaled_S_spectrum(Vphi, phidot, H, modes, modesdot, axis, k):
    """Power spectrum of S scaled with k^3/(2*pi^2)
    
    Assumes that k dimension is last.
    
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
    scaled_S_spectrum: array
                       Scaled spectrum of the isocurvature perturbation S
    """
    spectrum = Sspectrum(Vphi, phidot, H, modes, modesdot, axis)
    #Add extra dimensions to k if necessary
    scaled_spectrum = k**3/(2*np.pi**2) * spectrum
    return scaled_spectrum



def dprel_from_model(m, tix=None, kix=None):
    """Get the spectrum of delta Prel from a model instance.
    
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
    spectrum: array
              Array of values of the power spectrum of the relativistic pressure
              perturbation
    """
    components = utilities.components_from_model(m, tix, kix)
    result = deltaPrelspectrum(*components)
    
    return result
