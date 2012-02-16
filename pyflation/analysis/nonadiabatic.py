""" pyflation.analysis.nonadiabatic - Module to calculate relative pressure
perturbations.


"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


from __future__ import division
import numpy as np

import utilities

def soundspeeds(Vphi, phidot, H):
    """Sound speeds of the background fields
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
    
    Returns
    -------
    calphasq : array
               sound speed (squared) of the background fields
    
    Notes
    -----   
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
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    axis : integer
           Index of dimension to sum over (field dimension).
       
    All the arguments should have the same number of dimensions. Vphi and phidot
    should be arrays of the same size, but H should have a dimension of size 1 
    corresponding to the "field" dimension of the other variables.
    
    Returns
    -------
    csq : array_like
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
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    Returns
    -------
    Pdotalpha : array
                Derivative of the pressure for the background fields
                
    Notes
    -----
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
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
    
    axis : integer, optional
           Specifies which axis is the field dimension, default is the last one.
            
    Returns
    -------
    fullPdot : array
               The deriative of the pressure of the fields summed over the fields.
    """
    return np.sum(Pdots(Vphi, phidot, H), axis=axis)

def rhodots(phidot, H):
    """Derivative in e-fold time of the energy densities of the individual fields.
    
    Parameters
    ----------
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    Both arrays should have the same number of dimensions, but H should have a 
    dimension of size 1 corresponding to the field dimension of phidot.
    
    Returns
    -------
    rhodots : array
              The derivative of the energy densities of the fields
    """
    return -3*H**2*(phidot**2)

def fullrhodot(phidot, H, axis=-1):
    """Combined derivative in e-fold time of the energy density of the field.
    
    Parameters
    ----------
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
    
    axis : integer, optional
           Specifies which axis is the field dimension, default is the last one.
            
    Returns
    -------
    fullrhodot : array
                 The derivative of the energy density summed over the fields.
    """
    return np.sum(rhodots(phidot, H), axis=axis)



def deltarhosmatrix(Vphi, phidot, H, modes, modesdot, axis):
    """Matrix of the first order perturbed energy densities of the field components.
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
           
    modesdot : array_like
               Mode matrix of N-derivative of first order perturbations. 
               Component array should have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
    
    Returns
    -------
    result : array_like
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
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
           
    modesdot : array_like
               Mode matrix of N-derivative of first order perturbations. 
               Component array should have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
    
    Returns
    -------
    result : array_like
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
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    modesdot : array_like
               Mode matrix of N-derivative of first order perturbations. 
               Component array should have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
    
    Returns
    -------
    result : array
             Perturbed relative pressure of the fields.
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
    
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    modesdot : array_like
            Mode matrix of N-derivative of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    axis : integer
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
    
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    modesdot : array_like
               Mode matrix of N-derivative of first order perturbations. 
               Component array should have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
    
    Returns
    -------
    result : array
             Isocurvature perturbation S of the fields
    """
    
    dpnadmodes = deltaPnadmodes(Vphi, phidot, H, modes, modesdot, axis)
    Pdot = fullPdot(Vphi, phidot, H, axis)
    Pdot = np.expand_dims(Pdot, axis)
    result = dpnadmodes/Pdot       
    
    return result  

def deltarhospectrum(Vphi, phidot, H, modes, modesdot, axis):
    """Power spectrum of the perturbed energy density.
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    modesdot : array_like
               Mode matrix of N-derivative of first order perturbations. 
               Component array should have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
    
    Returns
    -------
    deltarhospectrum : array
                       Spectrum of the perturbed energy density
    """
    drhomodes = deltarhosmatrix(Vphi, phidot, H, modes, modesdot, axis)
    
    drhoI = np.sum(drhomodes, axis=axis)
    
    spectrum = utilities.makespectrum(drhoI, axis)
    
    return spectrum

def deltaPspectrum(Vphi, phidot, H, modes, modesdot, axis):
    """Power spectrum of the full perturbed relative pressure.
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    modesdot : array_like
               Mode matrix of N-derivative of first order perturbations. 
               Component array should have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
    
    Returns
    -------
    deltaPspectrum : array
                     Spectrum of the perturbed pressure
    """
    dPmodes = deltaPmatrix(Vphi, phidot, H, modes, modesdot, axis)
    
    dPI = np.sum(dPmodes, axis=axis)
    
    spectrum = utilities.makespectrum(dPI, axis)
    
    return spectrum

def deltaPrelspectrum(Vphi, phidot, H, modes, modesdot, axis):
    """Power spectrum of the full perturbed relative pressure.
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    modesdot : array_like
               Mode matrix of N-derivative of first order perturbations. 
               Component array should have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
    
    Returns
    -------
    deltaPrelspectrum : array
                       Spectrum of the perturbed relative pressure
    """
    dPrelI = deltaPrelmodes(Vphi, phidot, H, modes, modesdot, axis)
    
    spectrum = utilities.makespectrum(dPrelI, axis)
    
    return spectrum

def deltaPnadspectrum(Vphi, phidot, H, modes, modesdot, axis):
    """Power spectrum of the full perturbed non-adiabatic pressure.
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    modesdot : array_like
               Mode matrix of N-derivative of first order perturbations. 
               Component array should have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
    
    Returns
    -------
    deltaPnadspectrum : array
                        Spectrum of the non-adiabatic pressure perturbation
    """
    dPrelI = deltaPnadmodes(Vphi, phidot, H, modes, modesdot, axis)
    
    spectrum = utilities.makespectrum(dPrelI, axis)
    
    return spectrum

def Sspectrum(Vphi, phidot, H, modes, modesdot, axis):
    """Power spectrum of the isocurvature perturbation S.
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    modesdot : array_like
               Mode matrix of N-derivative of first order perturbations. 
               Component array should have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
    
    Returns
    -------
    Sspectrum : array
                Spectrum of the isocurvature perturbation S.
    """
    dSI = Smodes(Vphi, phidot, H, modes, modesdot, axis)
    
    spectrum = utilities.makespectrum(dSI, axis)
    
    return spectrum

def scaled_dPnad_spectrum(Vphi, phidot, H, modes, modesdot, axis, k):
    """Power spectrum of delta Pnad scaled with k^3/(2*pi^2)
    
    Assumes that k dimension is last.
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    modesdot : array_like
               Mode matrix of N-derivative of first order perturbations. 
               Component array should have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
          
    k : array
        The values of k to scale the result with.
    
    Returns
    -------
    scaled_dPnad_spectrum : array
                            Scaled spectrum of the non-adiabatic pressure 
                            perturation.
    """
    spectrum = deltaPnadspectrum(Vphi, phidot, H, modes, modesdot, axis)
    #Add extra dimensions to k if necessary
    scaled_spectrum = utilities.kscaling(k) * spectrum
    return scaled_spectrum

def scaled_dP_spectrum(Vphi, phidot, H, modes, modesdot, axis, k):
    """Power spectrum of delta P scaled with k^3/(2*pi^2)
    
    Assumes that k dimension is last.
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    modesdot : array_like
               Mode matrix of N-derivative of first order perturbations. 
               Component array should have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
    
    k : array
        The values of k to scale the result with.
    
    Returns
    -------
    scaled_dP_spectrum : array
                         Scaled spectrum of the perturbed pressure
    """
    spectrum = deltaPspectrum(Vphi, phidot, H, modes, modesdot, axis)
    #Add extra dimensions to k if necessary
    scaled_spectrum = utilities.kscaling(k) * spectrum
    return scaled_spectrum

def scaled_S_spectrum(Vphi, phidot, H, modes, modesdot, axis, k):
    """Power spectrum of S scaled with k^3/(2*pi^2)
    
    Assumes that k dimension is last.
    
    Parameters
    ----------
    Vphi : array_like
           First derivative of the potential with respect to the fields
          
    phidot : array_like
             First derivative of the field values with respect to efold number N.
            
    H : array_like
        The Hubble parameter
       
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    modesdot : array_like
               Mode matrix of N-derivative of first order perturbations. 
               Component array should have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
    
    k : array
        The values of k to scale the result with.
       
    Returns
    -------
    scaled_S_spectrum : array
                        Scaled spectrum of the isocurvature perturbation S
    """
    spectrum = Sspectrum(Vphi, phidot, H, modes, modesdot, axis)
    #Add extra dimensions to k if necessary
    scaled_spectrum = utilities.kscaling(k) * spectrum
    return scaled_spectrum

def scaled_S_from_model(m, tix=None, kix=None):
    """Return the spectrum of isocurvature perturbations :math:`\mathcal{P}_\mathcal{S}` 
    for each timestep and k mode.
    
    This is the scaled power spectrum which is related to the unscaled version by
    :math:`\mathcal{P}_\mathcal{S} = k^3/(2\pi^2) P_\mathcal{S}`. 
     
    Parameters
    ----------
    m : Cosmomodels instance
        Model class instance from which the yresult variable will be used to 
        calculate P_R.
    
    tix : integer, optional
          index of timestep at which to calculate, defaults to full range of steps.
         
    kix : integer, optional
          integer of k value at which to calculate, defaults to full range of ks.
            
    Returns
    -------
    scaled_S : array_like
               Array of spectrum values for all timesteps and k modes
    """
    if kix is None:
        kslice = slice(None)
    else:
        kslice = slice(kix, kix+1)
    components = utilities.components_from_model(m, tix, kix)
    spectrum = scaled_S_spectrum(*components, k=m.k[kslice]) 
    return spectrum

def slope_of_S_spectrum(scaled_S, k, kix=None, running=False):
    r"""Return the value of the slope of the k-scaled spectrum of S
    
    Parameters
    ----------
    scaled_S : array_like
               Power spectrum of isocurvature perturbations at a specific time
               This should be the *scaled* power spectrum i.e.
              
               .. math::
                   \rm{scaled_S} = k^3/(2\pi)^2 P_S,
                    
                   <S(k)S(k')> = (2\pi^3) \delta(k+k') P_S
               
               The array should be one-dimensional indexed by the k value.
           
    k : array_like
        Array of k values for which sPr has been calculated.
       
    kix : integer
          Index value of k for which to return spec_S.
         
    running : boolean, optional
              Whether running should be allowed or not. If true, a quadratic
              polynomial fit is made instead of linear and the value of the 
              running is returned along with the slope. Defaults to False.
       
    Returns
    -------
    spec_S : float
             The value of the spectral index of S at the requested k value and timestep.
             Normalised in the same way as n_s
             
             .. math::
                 \rm{spec_S} = 1 - d \log(\rm{scaled_S}) / d \log(k) 
                 
             evaluated at k[kix].
             
             This is calculated using a polynomial least squares fit with 
             numpy.polyfit. If running is True then a quadratic polynomial is fit,
             otherwise only a linear fit is made.
    
    running_S : float, present only if running = True
                If running=True the value of the derivative of the slope 
                at k[kix] is returned in a tuple along with spec_S.
    """ 
    
    result = utilities.spectral_index(scaled_S, k, kix, running)
    return result

def dprel_from_model(m, tix=None, kix=None):
    """Get the spectrum of delta Prel from a model instance.
    
    Parameters
    ----------
    m : Cosmomodels model instance
        The model instance with which to perform the calculation
       
    tix : integer
          Index for timestep at which to perform calculation. Default is to 
          calculate over all timesteps.
        
    kix : integer
          Index for k mode for which to perform the calculation. Default is to
          calculate over all k modes.
         
    Returns
    -------
    spectrum : array
               Array of values of the power spectrum of the relativistic pressure
               perturbation
    """
    components = utilities.components_from_model(m, tix, kix)
    result = deltaPrelspectrum(*components)
    
    return result

def S_alternate(phidot, Pphi_modes, axis):
    r"""Return the alternate spectrum of (first order) isocurvature perturbations P_S for each k.
    This is only available for a two field model and is given by:
    
    .. math::
        \rm{P_S} = (H/\dot{\sigma})^2 \langle\delta s \delta s*\rangle
            
    where 
    
    .. math::
        \dot{\sigma} = \sqrt{\dot{\phi}^2 + \dot{\chi}^2}
        
        \delta s = - \dot{\chi}/\dot{\sigma} \delta \phi
                     + \dot{\phi}/\dot{\sigma} \delta \chi
    
    This is the unscaled version :math:`P_S` which is related to the scaled version by
    :math:`\mathcal{P}_S = k^3/(2\pi^2) P_S`. 
    
    Parameters
    ----------
    phidot : array_like
             First derivative of the field values with respect to efold number N.
    
    Pphi_modes : array_like
                 Mode matrix of first order perturbation power spectrum given by
                 adiabatic.Pphi_matrix. Component array should have two dimensions 
                 of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
               
    Returns
    -------
    Pr : array_like, dtype: float64
         Array of Pr values for all timesteps and k modes
    """      
    nfields = Pphi_modes.shape[axis]
    if nfields != 2:
        raise ValueError("Only two field models can be used in this calculation.")
    
    phidot = np.atleast_1d(phidot)
    phidotsumsq = (np.sum(phidot**2, axis=axis))**2
    
    tslice = (slice(None),)*axis
    transform = np.ones_like(Pphi_modes)
    transform[tslice + (0,1)] = -1
    transform[tslice + (1,0)] = -1
    
    #Multiply mode matrix by corresponding phidot value
    phidotI = np.expand_dims(phidot[tslice + (slice(None,None,-1),)], axis)
    phidotJ = np.expand_dims(phidot[tslice + (slice(None,None,-1),)], axis+1)
    summatrix = phidotI*phidotJ*Pphi_modes*transform
    #Flatten mode matrix and sum over all nfield**2 values
    sumflat = np.sum(utilities.flattenmodematrix(summatrix, nfields, axis, axis+1), axis=1)
    #Divide by total sum of derivative terms
    Pr = (sumflat/phidotsumsq).astype(np.float)
    return Pr

def scaled_S_alternate_spectrum(phidot, Pphi_modes, axis, k):
    r"""Return the scaled alternate spectrum of (first order) isocurvature perturbations for each k.
    This is only available for a two field model and is given by:
    
    .. math::
        P_{\bar{S}} = (H/\dot{\sigma})^2 <\delta s \delta s*>
            
        \dot{\sigma} = \sqrt{\dot{\phi}^2 + \dot{\chi}^2} 
        
        \delta s = - \dot{\chi}/\dot{\sigma} \delta \phi
                     + \dot{\phi}/\dot{\sigma} \delta \chi
    
    This is the scaled version :math:`\mathcal{P}_{\bar{S}}` which is related to the unscaled version by
    :math:`\mathcal{P}_{\bar{S}} = k^3/(2\pi^2) P_{\bar{S}}`. 
    
    Parameters
    ----------
    phidot : array_like
             First derivative of the field values with respect to efold number N.
    
    Pphi_modes : array_like
                 Mode matrix of first order perturbation power spectrum given by
                 adiabatic.Pphi_matrix. Component array should have two dimensions 
                 of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
               
    Returns
    -------
    Pr : array_like, dtype: float64
         Array of Pr values for all timesteps and k modes
    """      
    spectrum = S_alternate(phidot, Pphi_modes, axis)
        #Add extra dimensions to k if necessary
    scaled_spectrum = utilities.kscaling(k) * spectrum
    return scaled_spectrum

