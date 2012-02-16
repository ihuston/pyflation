""" pyflation.analysis.adiabatic - Tools to calculate the spectra of adiabatic
perturbations

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


from __future__ import division
import numpy as np

import utilities
import nonadiabatic


def Pphi_modes(m):
    """Return the modes of the scalar perturbations P_{I,J}.
    
    This is a helper function which wraps the full calculation in Pphi_matrix
    and requires only the model as an argument. Provided for compatibility 
    with previous versions.
    
    Parameters
    ----------
    m : Cosmomodels instance
        Model class instance from which the yresult variable will be used to 
        calculate P_{I,J}.
       
    Returns
    -------
    Pphi_modes : array_like, dtype: float64
                 array of Pphi values flattened to have the same shape as
                 m.yresult[:,m.dps_ix].
    
    """
    #Get into mode matrix form, over first axis   
    mdp = utilities.getmodematrix(m.yresult, m.nfields, ix=1, ixslice=m.dps_ix)
    #Take tensor product of modes and conjugate, summing over second mode
    #index.
    mPphi = Pphi_matrix(mdp, axis=1)
    #Flatten back into vector form
    Pphi_modes = utilities.flattenmodematrix(mPphi, m.nfields, 1, 2) 
    return Pphi_modes

def Pphi_matrix(modes, axis):
    """Return the cross correlation of scalar perturbations P_{I,J}
    
    For multifield systems the full crossterm matrix is returned which 
    has shape nfields*nfields. 
    
    Parameters
    ----------
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
          
    Returns
    -------
    Pphi_matrix : array_like, dtype: float64
                  array of Pphi values with the same shape as input variable modes.
    """
    #Take tensor product of modes and conjugate, summing over second mode
    #index.
    mPphi = np.zeros_like(modes)
    #Do for loop as tensordot too memory expensive
    nfields=modes.shape[axis]
    if axis < 0:
        dims_to_skip = len(modes.shape) + axis
    else:
        dims_to_skip = axis
    preaxis = (slice(None),)*dims_to_skip
    for i in range(nfields):
        for j in range(nfields):
            for k in range(nfields):
                mPphi[preaxis+(i,j)] += modes[preaxis+(i,k)]*modes[preaxis+(j,k)].conj() 
    return mPphi


def findns(sPr, k, kix, running=False):
    r"""Return the value of n_s
    
    Parameters
    ----------
    sPr : array_like
          Power spectrum of scalar curvature perturbations at a specific time
          This should be the *scaled* power spectrum i.e.
    
          .. math::
               \rm{sPr} = k^3/(2*\pi)^2 P_\mathcal{R},
                
               <\mathcal{R}(k)\mathcal{R}(k')> = (2\pi^3) \delta(k+k') P_\mathcal{R}
               
          The array should be one-dimensional indexed by the k value.
           
    k : array_like
        Array of k values for which sPr has been calculated.
       
    kix : integer
          Index value of k for which to return n_s.
         
    running : boolean, optional
              Whether running should be allowed or not. If true, a quadratic
              polynomial fit is made instead of linear and the value of the 
              running alpha_s is returned along with n_s. Defaults to False.
       
    Returns
    -------
    n_s : float
          The value of the spectral index at the requested k value and timestep
          
          .. math::   
             \rm{n_s} = 1 - d \ln(\rm{sPr}) / d \ln(k) 
        
          evaluated at k[kix]
             
          This is calculated using a polynomial least squares fit with 
          numpy.polyfit. If running is True then a quadratic polynomial is fit,
          otherwise only a linear fit is made.
    
    alpha_s : float, present only if running = True
              If running=True the alpha_s value at k[kix] is returned in a 
              tuple along with n_s.
    """
    
    result = utilities.spectral_index(sPr, k, kix, running)
    return result

def findHorizoncrossings(m, factor=1):
    """Find horizon crossing for all ks
    
    Parameters
    ----------
    m : Cosmomodels instance
        Model for which to calculate horizon crossings

    factor : float
             Coefficient for which to calculate horizon crossings
             e.g. k=a*H*factor
             
    Returns
    -------
    hcross : array
             horizon crossing values for all ks.
    """
    return m.findallkcrossings(m.tresult, m.yresult[:,2], factor)
     
     
def Pr_spectrum(phidot, modes, axis):
    r"""Return the spectrum of (first order) curvature perturbations P_R1 for each k,
    given the first order perturbation modes.
    
    For a multifield model this is given by:
    
    .. math::
        P_\mathcal{R} = (\sum_K \dot{\phi_K}^2 )^{-2} \sum_{I,J} \dot{\phi_I} \dot{\phi_J} P_{IJ}
            
    where :math:`P_{IJ} = \sum_K \chi_{IK} \chi_{JK}`
    and :math:`\chi` are the mode matrix elements.  
    
    This is the unscaled version :math:`P_R` which is related to the scaled version by
    :math:`\mathcal{P}_\mathcal{R} = k^3/(2\pi^2) P_\mathcal{R}`.
    
    Parameters
    ----------
    phidot : array_like
             First derivative of the field values with respect to efold number N.
    
    modes : array_like
            Mode matrix of first order perturbations. Component array should
            have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
               
    Returns
    -------
    Pr : array_like, dtype: float64
         Array of Pr values for all timesteps and k modes
    """      
    phidot = np.atleast_1d(phidot)
    phidotsumsq = (np.sum(phidot**2, axis=axis))**2
    #Get mode matrix for Pphi_modes as nfield*nfield
    Pphimatrix = Pphi_matrix(modes, axis)
    #Multiply mode matrix by corresponding phidot value
    phidotI = np.expand_dims(phidot, axis)
    phidotJ = np.expand_dims(phidot, axis+1)
    summatrix = phidotI*phidotJ*Pphimatrix
    #Flatten mode matrix and sum over all nfield**2 values
    sumflat = np.sum(utilities.flattenmodematrix(summatrix, modes.shape[axis], axis, axis+1), axis=1)
    #Divide by total sum of derivative terms
    Pr = (sumflat/phidotsumsq).astype(np.float)
    return Pr

def Pr(m, tix=None, kix=None):
    r"""Return the spectrum of (first order) curvature perturbations P_R1 for a model m.
    
    For a multifield model this is given by:
    
    .. math::
        P_\mathcal{R} = (\sum_K \dot{\phi_K}^2 )^{-2} 
            \sum_{I,J} \dot{\phi_I} \dot{\phi_J} P_{IJ}
            
    where :math:`P_{IJ} = \sum_K \chi_{IK} \chi_{JK}`
    and :math:`\chi` are the mode matrix elements.  
    
    This is the unscaled version :math:`P_R` which is related to the scaled version by
    :math:`\mathcal{P}_\mathcal{R} = k^3/(2\pi^2) P_\mathcal{R}`.
    
    This function is a wrapper function which only requires the model as a parameter.
    The full calculation is done in the Pr_spectrum function in the 
    pyflation.analysis.adiabatic module.
    
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
    Pr : array_like, dtype: float64
         Array of Pr values for requested timesteps and kmodes
        
    See also
    --------
    pyflation.analysis.adiabatic.Pr_spectrum function
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
        
    phidot = np.float64(m.yresult[tslice,m.phidots_ix,kslice]) #bg phidot
    #Get into mode matrix form, over first axis   
    modes = utilities.getmodematrix(m.yresult[tslice,...,kslice], m.nfields, ix=1, ixslice=m.dps_ix)
    axis = 1
    Pr = Pr_spectrum(phidot, modes, axis)
    return Pr
    

def scaled_Pr(m, tix=None, kix=None):
    r"""Return the spectrum of (first order) curvature perturbations 
    :math:`\mathcal{P}_\mathcal{R}` for each timestep and k mode.
    
    This is the scaled power spectrum which is related to the unscaled version by
    :math:`\mathcal{P}_\mathcal{R} = k^3/(2\pi^2) P_\mathcal{R}`. 
     
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
    calPr : array_like
            Array of Pr values for all timesteps and k modes
    """
    return utilities.kscaling(m.k, kix) * Pr(m, tix, kix)           

def Pzeta(m, tix=None, kix=None):
    r"""Return the spectrum of (first order) curvature perturbations P_zeta for each k.
    
    For a multifield model this is given by:
    
    .. math::
        \rm{Pzeta} = \mathcal{P}_{\delta \rho} / (\rho^\dagger)^2
    
    This is the unscaled version P_zeta which is related to the scaled version by
    :math:`\mathcal{P}_\zeta = k^3/(2\pi^2) P_\zeta`.
    
    Parameters
    ----------
    m : Cosmomodels instance
        model containing yresult with which to calculate spectrum
    
    tix : integer, optional
          index of timestep at which to calculate, defaults to full range of steps.
         
    kix : integer, optional
          integer of k value at which to calculate, defaults to full range of ks.
    
    Returns
    -------
    Pzeta : array_like, dtype: float64
            Array of Pzeta values for all timesteps and k modes
    """      
    Vphi,phidot,H,modes,modesdot,axis = utilities.components_from_model(m, tix, kix)
    Pzeta = Pzeta_spectrum(Vphi, phidot, H, modes, modesdot, axis)
    return Pzeta

def Pzeta_spectrum(Vphi, phidot, H, modes, modesdot, axis):
    r"""Return the spectrum of (first order) curvature perturbations P_zeta for each k.
    
    For a multifield model this is given by:
    
    .. math::
        \rm{Pzeta} = \mathcal{P}_{\delta \rho} / (\rho^\dagger)^2
    
    This is the unscaled version P_zeta which is related to the scaled version by
    :math:`\mathcal{P}_\zeta = k^3/(2\pi^2) P_\zeta`. 
    
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
    Pzeta_spectrum : array_like, dtype: float64
                     Array of Pzeta values
    """      
    Vphi,phidot,H,modes,modesdot,axis = utilities.correct_shapes(Vphi, phidot, H, 
                                                                 modes, modesdot, axis)
    rhodot = nonadiabatic.fullrhodot(phidot, H, axis)
    drhospectrum = nonadiabatic.deltarhospectrum(Vphi, phidot, H, modes, modesdot, axis)
    Pzeta = rhodot**(-2.0) * drhospectrum
    return Pzeta

def scaled_Pzeta(m, tix=None, kix=None):
    r"""Return the spectrum of scaled (first order) curvature perturbations 
    :math:`\mathcal{P}_\zeta` for each timestep and k mode.
    
    This is the scaled power spectrum which is related to the unscaled version by
    :math:`\mathcal{P}_\zeta = k^3/(2\pi^2) P_\zeta`. 
    
    Parameters
    ----------
    m : Cosmomodels instance
        model containing yresult with which to calculate spectrum
        
    tix : integer, optional
          index of timestep at which to calculate, defaults to full range of steps.
         
    kix : integer, optional
          integer of k value at which to calculate, defaults to full range of ks.
    
    
    Returns
    -------
    scaled_Pzeta : array_like
                   Array of Pzeta values for all timesteps and k modes
    """
    return utilities.kscaling(m.k, kix) * Pzeta(m, tix, kix)      

def Pr_spectrum_from_Pphimodes(phidot, Pphi_modes, axis):
    r"""Return the spectrum of (first order) curvature perturbations P_R1 for each k.
    
    For a multifield model this is given by:
    
    .. math::
        \rm{Pr} = (\sum_K \dot{\phi_K}^2 )^{-2} 
            \sum_{I,J} \dot{\phi_I} \dot{\phi_J} P_{IJ}
            
    where :math:`P_{IJ} = \sum_K \chi_{IK} \chi_{JK}`
    and :math:`\chi` are the mode matrix elements.  
    
    This is the unscaled version :math:`P_\mathcal{R}` which is related to the 
    scaled version by :math:`\mathcal{P}_\mathcal{R} = k^3/(2pi^2) P_\mathcal{R}`.  
    
    Parameters
    ----------
    phidot : array_like
             First derivative of the field values with respect to efold number N.
    
    Pphi_modes : array_like
                 Mode matrix of first order perturbations spectra. Component array should
                 have two dimensions of length nfields.
    
    axis : integer
           Specifies which axis is first in mode matrix, e.g. if modes has shape
           (100,3,3,10) with nfields=3, then axis=1. The two mode matrix axes are
           assumed to be beside each other so (100,3,10,3) would not be valid.
               
    Returns
    -------
    Pr : array_like, dtype: float64
         Array of Pr values for all timesteps and k modes
    """      
    phidot = np.atleast_1d(phidot)
    phidotsumsq = (np.sum(phidot**2, axis=axis))**2
    #Multiply mode matrix by corresponding phidot value
    phidotI = np.expand_dims(phidot, axis)
    phidotJ = np.expand_dims(phidot, axis+1)
    summatrix = phidotI*phidotJ*Pphi_modes
    #Flatten mode matrix and sum over all nfield**2 values
    sumflat = np.sum(utilities.flattenmodematrix(summatrix, Pphi_modes.shape[axis], axis, axis+1), axis=1)
    #Divide by total sum of derivative terms
    Pr = (sumflat/phidotsumsq).astype(np.float)
    return Pr

