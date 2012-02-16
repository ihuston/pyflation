# -*- coding: utf-8 -*-
"""cmpotentials.py - Cosmological potentials for cosmomodels.py

Provides functions which can be used with cosmomodels.py. 
Default parameter values are included but can also be 
specified as a dictionary.

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


from __future__ import division
import numpy as np

def msqphisq(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for V=1/2 m^2 phi^2
    where m is the mass of the inflaton field.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
        
    Notes
    -----
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    """
    #Check if mass is specified in params
    if params is not None and "mass" in params:
        m = params["mass"]
    else:
        #Use WMAP value of mass (in Mpl)
        m = 6.3267e-6
        
    if len(y.shape)>1:
        y = y[:,0]
        
    # The shape of the potentials is important to be consistent with the
    # multifield case. The following shapes should be used for a single field
    # model:
    #
    # U : scalar (use np.asscalar)
    # dUdphi : 1d vector (use np.atleast_1d)
    # d2Udphi2 : 2d array (use np.atleast_2d)
    # d3Udphi3 : 3d array (use np.atleast_3d)
    
    #Use inflaton mass
    mass2 = m**2
    #potential U = 1/2 m^2 \phi^2
    U = np.asscalar(0.5*(mass2)*(y[0]**2))
    #deriv of potential wrt \phi
    dUdphi =  np.atleast_1d((mass2)*y[0])
    #2nd deriv
    d2Udphi2 = np.atleast_2d(mass2)
    #3rd deriv
    d3Udphi3 = np.atleast_3d(0)
    
    return U, dUdphi, d2Udphi2, d3Udphi3

def lambdaphi4(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for V=1/4 lambda phi^4
    for a specified lambda.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameter "lambda" which specifies lambda
             above.
    
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
    
    Notes
    -----
    lambda can be specified in the dictionary params or otherwise
    it defaults to the value as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    """
    #Check if mass is specified in params
    if params is not None and "lambda" in params:
        l = params["lambda"]
    else:
        #Use WMAP value of lambda
        l = 1.5506e-13 
    if len(y.shape)>1:
        y = y[:,0]
    
    # The shape of the potentials is important to be consistent with the
    # multifield case. The following shapes should be used for a single field
    # model:
    #
    # U : scalar (use np.asscalar)
    # dUdphi : 1d vector (use np.atleast_1d)
    # d2Udphi2 : 2d array (use np.atleast_2d)
    # d3Udphi3 : 3d array (use np.atleast_3d)
    
    #potential U = 1/4 l \phi^4
    U = np.asscalar(0.25*l*(y[0]**4))
    #deriv of potential wrt \phi
    dUdphi =  np.atleast_1d(l*(y[0]**3))
    #2nd deriv
    d2Udphi2 = np.atleast_2d(3*l*(y[0]**2))
    #3rd deriv
    d3Udphi3 = np.atleast_3d(6*l*(y[0]))
    
    return U, dUdphi, d2Udphi2, d3Udphi3
    
def linde(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for Linde potential
    V = -m^2/2 \phi^2 +\lambda/4 \phi^4 + m^4/4lambda
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameters "mass" and "lambda" which specifies 
             the variables.
             
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
        
    Notes
    -----
    lambda can be specified in the dictionary params or otherwise
    it defaults to the value as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    
    mass can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    """
    #Check if mass is specified in params
    if params is not None and "mass" in params:
        m = params["mass"]
    else:
        #Use Salopek et al value of mass (in Mpl)
        m = 5e-8
    #Use inflaton mass
    mass2 = m**2
    #Check if mass is specified in params
    if params is not None and "lambda" in params:
        l = params["lambda"]
    else:
        #Use WMAP value of lambda
        #l = 1.5506e-13
        l = 1.55009e-13 
    
    if len(y.shape)>1:
        y = y[:,0]
        
    # The shape of the potentials is important to be consistent with the
    # multifield case. The following shapes should be used for a single field
    # model:
    #
    # U : scalar (use np.asscalar)
    # dUdphi : 1d vector (use np.atleast_1d)
    # d2Udphi2 : 2d array (use np.atleast_2d)
    # d3Udphi3 : 3d array (use np.atleast_3d)
    
    U = np.asscalar(-0.5*(mass2)*(y[0]**2) + 0.25*l*(y[0]**4) + (m**4)/(4*l))
    #deriv of potential wrt \phi
    dUdphi =  np.atleast_1d(-(mass2)*y[0] + l*(y[0]**3))
    #2nd deriv
    d2Udphi2 = np.atleast_2d(-mass2 + 3*l*(y[0]**2))
    #3rd deriv
    d3Udphi3 = np.atleast_3d(6*l*(y[0]))
    
    return U, dUdphi, d2Udphi2, d3Udphi3
    
def hybrid2and4(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for hybrid potential
    V = -m^2/2 \phi^2 +\lambda/4 \phi^4 
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameters "mass" and "lambda" which specifies 
             the variables.
    
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
        
    Notes
    -----     
    lambda can be specified in the dictionary params or otherwise
    it defaults to the value as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    
    mass can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    """
    #Check if mass is specified in params
    if params is not None and "mass" in params:
        m = params["mass"]
    else:
        #Use Salopek et al value of mass (in Mpl)
        m = 5e-8
    #Use inflaton mass
    mass2 = m**2
    #Check if mass is specified in params
    if params is not None and "lambda" in params:
        l = params["lambda"]
    else:
        #Use WMAP value of lambda
        l = 1.55123e-13
        
    
    if len(y.shape)>1:
        y = y[:,0]
        
    # The shape of the potentials is important to be consistent with the
    # multifield case. The following shapes should be used for a single field
    # model:
    #
    # U : scalar (use np.asscalar)
    # dUdphi : 1d vector (use np.atleast_1d)
    # d2Udphi2 : 2d array (use np.atleast_2d)
    # d3Udphi3 : 3d array (use np.atleast_3d)
    
    U = np.asscalar(0.5*(mass2)*(y[0]**2) + 0.25*l*(y[0]**4))
    #deriv of potential wrt \phi
    dUdphi =  np.atleast_1d((mass2)*y[0] + l*(y[0]**3))
    #2nd deriv
    d2Udphi2 = np.atleast_2d(mass2 + 3*l*(y[0]**2))
    #3rd deriv
    d3Udphi3 = np.atleast_3d(6*l*(y[0]))
    
    return U, dUdphi, d2Udphi2, d3Udphi3
    
def phi2over3(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for V= sigma phi^(2/3)
    for a specified sigma.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameter "sigma" which specifies lambda
             above.
    
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
        
    Notes
    -----  
    sigma can be specified in the dictionary params or otherwise
    it defaults to the value as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    """
    #Check if mass is specified in params
    if params is not None and "sigma" in params:
        s = params["sigma"]
    else:
        #Use WMAP value of lambda
        s = 3.81686e-10 #Unit Mpl^{10/3}
    if len(y.shape)>1:
        y = y[:,0]
        
    # The shape of the potentials is important to be consistent with the
    # multifield case. The following shapes should be used for a single field
    # model:
    #
    # U : scalar (use np.asscalar)
    # dUdphi : 1d vector (use np.atleast_1d)
    # d2Udphi2 : 2d array (use np.atleast_2d)
    # d3Udphi3 : 3d array (use np.atleast_3d)
    
    #potential U = 1/4 s \phi^4
    U = np.asscalar(s*(y[0]**(2.0/3)))
    #deriv of potential wrt \phi
    dUdphi =  np.atleast_1d((2.0/3)*s*(y[0]**(-1.0/3)))
    #2nd deriv
    d2Udphi2 = np.atleast_2d(-(2.0/9)*s*(y[0]**(-4.0/3)))
    #3rd deriv
    d3Udphi3 = np.atleast_3d((8.0/27)*s*(y[0]**(-7.0/3)))
    
    return U, dUdphi, d2Udphi2, d3Udphi3
    
def msqphisq_withV0(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for V=1/2 m^2 phi^2 + V0
    where m is the mass of the inflaton field.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
    
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
        
    Notes
    -----  
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    """
    #Check if mass is specified in params
    if params is not None and "mass" in params:
        m = params["mass"]
    else:
        #Use WMAP value of mass (in Mpl)
        m = 1.7403553e-06
    if params is not None and "V0" in params:
        V0 = params["V0"]
    else:
        V0 = 5e-10 # Units Mpl^4
    
    if len(y.shape)>1:
        y = y[:,0]
        
    # The shape of the potentials is important to be consistent with the
    # multifield case. The following shapes should be used for a single field
    # model:
    #
    # U : scalar (use np.asscalar)
    # dUdphi : 1d vector (use np.atleast_1d)
    # d2Udphi2 : 2d array (use np.atleast_2d)
    # d3Udphi3 : 3d array (use np.atleast_3d)
    
    #Use inflaton mass
    mass2 = m**2
    #potential U = 1/2 m^2 \phi^2
    U = np.asscalar(0.5*(mass2)*(y[0]**2) + V0)
    #deriv of potential wrt \phi
    dUdphi =  np.atleast_1d((mass2)*y[0])
    #2nd deriv
    d2Udphi2 = np.atleast_2d(mass2)
    #3rd deriv
    d3Udphi3 = np.atleast_3d(0)
    
    return U, dUdphi, d2Udphi2, d3Udphi3
    
def step_potential(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for 
    V=1/2 m^2 phi^2 ( 1 + c*tanh((phi-phi_s) / d)
    where m is the mass of the inflaton field and c, d and phi_s are provided.
    Form is taken from Chen etal. arxiv:0801.3295.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
    
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
        
    Notes
    -----  
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    """
    #Check if mass is specified in params
    if params is not None and "mass" in params:
        m = params["mass"]
    else:
        #Use WMAP value of mass (in Mpl)
        m = 6.3267e-6
    if params is not None:
        c = params.get("c", 0.0018)
        d = params.get("d", 0.022) #Units of Mpl
        phi_s = params.get("phi_s", 14.84) #Units of Mpl
    else:
        c = 0.0018
        d = 0.022
        phi_s = 14.84
    
    if len(y.shape)>1:
        y = y[:,0]
        
    # The shape of the potentials is important to be consistent with the
    # multifield case. The following shapes should be used for a single field
    # model:
    #
    # U : scalar (use np.asscalar)
    # dUdphi : 1d vector (use np.atleast_1d)
    # d2Udphi2 : 2d array (use np.atleast_2d)
    # d3Udphi3 : 3d array (use np.atleast_3d)
    
    #Use inflaton mass
    mass2 = m**2
    #potential U = 1/2 m^2 \phi^2
    
    phisq = y[0]**2
    
    phiterm = (y[0]-phi_s)/d
    s = 1/np.cosh(phiterm)
    t = np.tanh(phiterm)
    
    U = np.asscalar(0.5*(mass2)*(y[0]**2) * (1 + c * (t - 1)))
    #deriv of potential wrt \phi
    dUdphi =  np.atleast_1d((mass2)*y[0] * (1 + c*(t-1)) + c * mass2 * phisq * s**2 / (2*d))
    #2nd deriv
    d2Udphi2 = np.atleast_2d(0.5*mass2*(4*c*y[0]*s**2/d - 2*c*phisq*s**2*t/(d**2) + 2*(1+c*(t-1))))
    #3rd deriv
    d3Udphi3 = np.atleast_3d(0.5*mass2*(6*c*s**2/d - 12*c*y[0]*s**2*t/(d**2) 
                          + c*phisq*(-2*s**4/(d**3) + 4*s**2*t**2/(d**3))))
    
    return U, dUdphi, d2Udphi2, d3Udphi3

def bump_potential(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for 
    V=1/2 m^2 phi^2 ( 1 + c*sech((phi-phi_b) / d)
    where m is the mass of the inflaton field and c, d and phi_b are provided.
    Form is taken from Chen etal. arxiv:0801.3295.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
            
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
        
    Notes
    -----  
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    """
    #Check if mass is specified in params
    if params is not None and "mass" in params:
        m = params["mass"]
    else:
        #Use WMAP value of mass (in Mpl)
        m = 6.3267e-6
    if params is not None:
        c = params.get("c", 0.0005)
        d = params.get("d", 0.01) #Units of Mpl
        phi_b = params.get("phi_b", 14.84) #Units of Mpl
    else:
        c = 0.0005
        d = 0.01
        phi_b = 14.84
    
    #Use inflaton mass
    mass2 = m**2
    #potential U = 1/2 m^2 \phi^2
    
    if len(y.shape)>1:
        y = y[:,0]
        
    # The shape of the potentials is important to be consistent with the
    # multifield case. The following shapes should be used for a single field
    # model:
    #
    # U : scalar (use np.asscalar)
    # dUdphi : 1d vector (use np.atleast_1d)
    # d2Udphi2 : 2d array (use np.atleast_2d)
    # d3Udphi3 : 3d array (use np.atleast_3d)
    
    phisq = y[0]**2
    
    phiterm = (y[0]-phi_b)/d
    s = 1/np.cosh(phiterm)
    t = np.tanh(phiterm)
    
    U = np.asscalar(0.5*(mass2)*(y[0]**2) * (1 + c * s))
    #deriv of potential wrt \phi
    dUdphi =  np.atleast_1d((mass2)*y[0] * (1 + c*s) - c * mass2 * phisq * s*t / (2*d))
    #2nd deriv
    d2Udphi2 = np.atleast_2d(0.5*mass2*(-4*c*y[0]*s*t/d + c*phisq*(-s**3/(d**2) + s*(t**2)/(d**2)) + 2*(1+c*s)))
    #3rd deriv
    d3Udphi3 = np.atleast_3d(0.5*mass2*(-6*c*s*t/d + 6*c*y[0]*(-s**3/(d**2) + s*(t**2)/(d**2)) 
                          + c*phisq*(5*s**3*t/(d**3) - s*t**3/(d**3))))
    
    return U, dUdphi, d2Udphi2, d3Udphi3

def resonance(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for 
    V=1/2 m^2 phi^2 ( 1 + c*sin(phi / d) )
    where m is the mass of the inflaton field and c, d and phi_b are provided.
    Form is taken from Chen etal. arxiv:0801.3295.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above, 
             and the parameters "c" and "d" which tune the oscillation.
    
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
        
    Notes
    -----  
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    """
    #Check if mass is specified in params
    if params is not None and "mass" in params:
        m = params["mass"]
    else:
        #Use WMAP value of mass (in Mpl)
        m = 6.3267e-6
    if params is not None:
        c = params.get("c", 5e-7)
        d = params.get("d", 0.0007) #Units of Mpl
    else:
        c = 5e-7
        d = 0.0007
    
    #Use inflaton mass
    mass2 = m**2
    #potential U = 1/2 m^2 \phi^2
    
    if len(y.shape)>1:
        y = y[:,0]
        
    # The shape of the potentials is important to be consistent with the
    # multifield case. The following shapes should be used for a single field
    # model:
    #
    # U : scalar (use np.asscalar)
    # dUdphi : 1d vector (use np.atleast_1d)
    # d2Udphi2 : 2d array (use np.atleast_2d)
    # d3Udphi3 : 3d array (use np.atleast_3d)
    
    phi = y[0]
    phisq = phi**2
    
    phiterm = phi/d
    sphi = np.sin(phiterm)
    cphi = np.cos(phiterm)
    
    U = np.asscalar(0.5*(mass2)*(phisq) * (1 + c * sphi))
    #deriv of potential wrt \phi
    dUdphi =  np.atleast_1d((mass2)*phi * (1 + c*sphi) + c * mass2 * phisq * cphi / (2*d))
    #2nd deriv
    d2Udphi2 = np.atleast_2d(mass2*((1+c*sphi) + 2*c/d * cphi * phi))
    #3rd deriv
    d3Udphi3 = np.atleast_3d(mass2*(3*c/d*cphi -3*c/d**2*sphi * phi -0.5*c/d**3 *cphi * phisq))
    
    return U, dUdphi, d2Udphi2, d3Udphi3

def bump_nothirdderiv(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for 
    V=1/2 m^2 phi^2 ( 1 + c*sech((phi-phi_b) / d)
    where m is the mass of the inflaton field and c, d and phi_b are provided.
    Form is taken from Chen etal. arxiv:0801.3295.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
    
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
        
    Notes
    -----  
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    """
    #Check if mass is specified in params
    if params is not None and "mass" in params:
        m = params["mass"]
    else:
        #Use WMAP value of mass (in Mpl)
        m = 6.3267e-6
    if params is not None:
        c = params.get("c", 0.0005)
        d = params.get("d", 0.01) #Units of Mpl
        phi_b = params.get("phi_b", 14.84) #Units of Mpl
    else:
        c = 0.0005
        d = 0.01
        phi_b = 14.84
    
    #Use inflaton mass
    mass2 = m**2
    #potential U = 1/2 m^2 \phi^2
    if len(y.shape)>1:
        y = y[:,0]
        
    # The shape of the potentials is important to be consistent with the
    # multifield case. The following shapes should be used for a single field
    # model:
    #
    # U : scalar (use np.asscalar)
    # dUdphi : 1d vector (use np.atleast_1d)
    # d2Udphi2 : 2d array (use np.atleast_2d)
    # d3Udphi3 : 3d array (use np.atleast_3d)
        
    phisq = y[0]**2
    
    phiterm = (y[0]-phi_b)/d
    s = 1/np.cosh(phiterm)
    t = np.tanh(phiterm)
    
    U = np.asscalar(0.5*(mass2)*(y[0]**2) * (1 + c * s))
    #deriv of potential wrt \phi
    dUdphi =  np.atleast_1d((mass2)*y[0] * (1 + c*s) - c * mass2 * phisq * s*t / (2*d))
    #2nd deriv
    d2Udphi2 = np.atleast_2d(0.5*mass2*(-4*c*y[0]*s*t/d + c*phisq*(-s**3/(d**2) + s*(t**2)/(d**2)) + 2*(1+c*s)))
    #3rd deriv
    d3Udphi3 = np.atleast_3d(0.0)
    
    return U, dUdphi, d2Udphi2, d3Udphi3

def hybridquadratic(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for 
    V = 1/2 m1^2 phi^2 + 1/2 m2^2 chi^2
    where m1 and m2 are the masses of the fields. Needs nfields=2.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameters "m1" and "m2" specified above.
            
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
        
    """
    
    #Check if mass is specified in params
    if params:
        m1 = params.get("m1", 1.395464769e-6)
        m2 = params.get("m2", 9.768253382e-6)
    else:
        m1 = 1.395464769e-6
        m2 = 9.768253382e-6
        
    if len(y.shape)>1:
        y = y[:,0]
        
    #Use inflaton mass
    mass2 = np.array([m1, m2])**2
    #potential U = 1/2 m^2 \phi^2
    U = np.asscalar(0.5*(m1**2*y[0]**2 + m2**2*y[2]**2))
    #deriv of potential wrt \phi
    dUdphi = mass2*np.array([y[0],y[2]])
    #2nd deriv
    d2Udphi2 = mass2*np.eye(2)
    #3rd deriv
    d3Udphi3 = np.zeros((2,2,2))
    
    return U, dUdphi, d2Udphi2, d3Udphi3

def ridge_twofield(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for V=V0 - g phi - 1/2 m^2 chi^2
    where g is a parameter and m is the mass of the chi field. Needs nfields=2.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameters "V0", "g", "m".
            
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
        
    """
    
    #Check if mass is specified in params
    if params:
        g = params.get("g", 1e-5)
        m = params.get("m", 12e-5)
        V0 = params.get("V0", 1)
    else:
        g = 1e-5
        m = 12e-5
        V0 = 1
        
    if len(y.shape)>1:
        y = y[:,0]
        
    #potential U = 1/2 m^2 \phi^2
    U = np.asscalar(V0 - g*y[0] - 0.5*m**2*y[2]**2)
    #deriv of potential wrt \phi
    dUdphi = np.array([-g, -m**2 * y[2]])
    #2nd deriv
    d2Udphi2 = np.array([[0,0], [0,-m**2]])
    #3rd deriv
    d3Udphi3 = np.zeros((2,2,2))
    
    return U, dUdphi, d2Udphi2, d3Udphi3

def nflation(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for 
    V = \sum_\alpha 1/2 m^2 \phi_\alpha^2
    where m is the mass of each of the fields.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
             The number of fields is specified through "nfields".
            
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
        
    Notes
    -----  
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    """
    
    #Check if mass is specified in params
    if params is not None and "mass" in params:
        m = params["mass"]
    else:
        #Use WMAP value of mass (in Mpl)
        m = 6.3267e-6
    
    nfields = params["nfields"]    
    
    if len(y.shape)>1:
        y = y[:,0]
        
    phis_ix = slice(0,nfields*2,2)
    
    #Use inflaton mass
    mass2 = m**2
    #potential U = 1/2 m^2 \phi^2
    U = np.sum(0.5*(mass2)*(y[phis_ix]**2))
    #deriv of potential wrt \phi
    dUdphi =  (mass2)*y[phis_ix]
    #2nd deriv
    d2Udphi2 = mass2*np.eye(nfields, dtype=np.complex128)
    #3rd deriv
    d3Udphi3 = None
    
    return U, dUdphi, d2Udphi2, d3Udphi3

def quartictwofield(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for 
    V= 1/2(m1^2 \phi^2 + 1/2 l1 \phi^4 + m2^2 \chi^2 + 1/2 l2 \chi^4)
    where m1, m2, l1, l2 are parameters. Needs nfields=2.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values in this case should
             hold the parameters "m1", "m2", "l1", "l2", as specified above.
            
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
             
    """
    
    #Check if mass is specified in params
    if params:
        m1 = params.get("m1", 5e-6)
        m2 = params.get("m2", 5e-8)
        l1 = params.get("l1", 5e-10)
        l2 = params.get("l2", 5e-14)
    else:
        m1 = 5e-6
        m2 = 5e-8
        
    if len(y.shape)>1:
        y = y[:,0]
        
    #potential U = 1/2 m^2 \phi^2
    U = np.asscalar(0.5*(m1**2*y[0]**2 + 0.5*l1*y[0]**4 + m2**2*y[2]**2 + 0.5*l2*y[2]**4))
    #deriv of potential wrt \phi
    dUdphi = np.array([m1**2*y[0] + l1*y[0]**3, m2**2*y[2] + l2*y[2]**3])
    #2nd deriv
    d2Udphi2 = np.eye(2)*np.array([m1**2 + 3*l1*y[0]**2, m2**2 + 3*l2*y[2]**2])
    #3rd deriv
    d3Udphi3 = None
    
    return U, dUdphi, d2Udphi2, d3Udphi3

def hybridquartic(y, params=None):
    """Return the potential and its first three derivatives for the hybrid
    quartic model.
    
    The potential is given by
    
    .. math:: 
        V = \Lambda^4 [ (1-\chi^2/v^2)^2 + \phi^2/\mu^2 
                    + 2\phi^2\chi^2/(\phi_c^2 v^2) ]
                    
    where the parameter are :math:`\Lambda, v, \mu and \phi_c`. Needs nfields=2.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values labelled "lambda" , "v", 
             "mu", "phi_c".
            
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
        Tuple of the potential and its first three derivatives.
             
    """
    
    #Check if mass is specified in params
    if params:
        l = params.get("lambda", 2.3644e-6)
        v = params.get("v", 0.1)
        mu = params.get("mu", 1e3)
        phi_c = params.get("phi_c", 0.01)
    else:
        l = 2.3644e-6
        v = 0.1
        mu = 1e3
        phi_c = 0.01
        
    if len(y.shape)>1:
        y = y[:,0]
        
    phi = y[0]
    chi = y[2]
    
    l4 = l**4
    phicv2 = (phi_c*v)**2
    
    #potential U = 1/2 m^2 \phi^2
    U = np.asscalar(l4 *((1-chi**2/v**2)**2 + phi**2/mu**2 + 2*(phi*chi)**2/phicv2))
    #deriv of potential wrt \phi
    dUdphi = l4*np.array([2*phi/mu**2 + 4*phi*chi**2/phicv2, 
                            -4*chi/v**2 * (1-chi**2/v**2) + 4*phi**2*chi/phicv2])
    #2nd deriv
    d2Udphi2 = l4*np.array([[2/mu**2 + 4*chi**2/phicv2, # V phi phi 
                             8*phi*chi/phicv2],         # V phi chi
                            [8*phi*chi/phicv2,          # V chi phi
                             -4/v**2 * (1-3*chi**2/v**2) + 4*phi**2/phicv2]]) # V chi chi
    #3rd deriv Not set as not used in first order calculation
    d3Udphi3 = np.zeros((2,2,2))
    
    return U, dUdphi, d2Udphi2, d3Udphi3

def inflection(y, params=None):
    r"""Return the potential and its first three derivatives for an inflection
    point model.
    
    The potential is given by
    
    .. math:: 
        V = V_0 + 0.5 m^2 \phi^2 + g \chi + 1/6 \lambda \chi^3 + \lambda/(8 r) \chi^4
    
    where :math:`V_0 = 0.75gr + \lambda/24 r^3 + g/(4r^3)`
    and the parameters are :math:`\lambda, m, g and r`. Needs nfields=2.
    
    Parameters
    ----------
    y : array
        Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params : dict
             Dictionary of parameter values labelled "lambda" , "m", "g", "r".
    
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3 : tuple of arrays
         Tuple of the potential and its first three derivatives.
         
    """
    #Check if mass is specified in params
    if params:
        l = params.get("lambda", 3e3)
        g = params.get("g", 3e-2)
        r = params.get("r", 0.14)
        m = params.get("m", 1.0)
    else:
        l = 3e3
        g = 3e-2
        r = 0.14
        m = 1.0
                
    if len(y.shape)>1:
        y = y[:,0]
        
    V_0 = 0.75*g*r + l/24.0 * r**3 
    phi = y[0]
    chi = y[2]
    
    #potential U = 1/2 m^2 \phi^2
    U = np.asscalar(V_0 + 0.5*m**2*phi**2 + g*chi + 1/6.0 * l * chi**3
                    + (g/(4*r**3) + l/(8*r)) * chi**4)
    #deriv of potential wrt \phi
    dUdphi = np.array([m**2*phi, g + 0.5 * l * chi**2 + (g/(2*r**3) + l/(2*r)) * chi**3])
    #2nd deriv
    d2Udphi2 = np.array([[m**2, # V phi phi 
                          0.0],         # V phi chi
                         [0.0,          # V chi phi
                          l*chi + 3/2*(g/r**3 + l/r) * chi**2]]) # V chi chi
    #3rd deriv Not set as not used in first order calculation
    d3Udphi3 = np.zeros((2,2,2))
    
    return U, dUdphi, d2Udphi2, d3Udphi3

def hilltopaxion(y, params=None):
    r"""Return the potential and its first three derivatives for a hilltop axion
    model.
    
    The potential is given by
    
    .. math:: 
        V = 0.5 m^2 \varphi^2 + \Lambda^4 (1 - \cos(2\pi\chi/f))
        
    where the parameters are \Lambda, m, and f . Needs nfields=2.
    
    Parameters
    ----------
    y: array
       Array of variables with background phi as y[0]
       If you want to specify a vector of phi values, make sure
       that the first index still runs over the different 
       variables, using newaxis if necessary.
    
    params: dict
            Dictionary of parameter values labelled "Lambda" , "m", "f".
            
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3: tuple of arrays
        Tuple of the potential and its first three derivatives.
             
    """
    
    #Check if mass is specified in params
    if params:
        l = params.get("Lambda", np.sqrt(6e-6/(4*np.pi)))
        f = params.get("f", 1.0)
        m = params.get("m", 6e-6)
    else:
        l = np.sqrt(6e-6/(4*np.pi))
        f = 1.0
        m = 6e-6
                
    if len(y.shape)>1:
        y = y[:,0]
        
     
    phi = y[0]
    chi = y[2]
    
    twopif = 2*np.pi/f
    
    #potential U = 1/2 m^2 \phi^2
    U = np.asscalar(0.5*m**2*phi**2 + l**4*(1 - np.cos(twopif*chi)))
    #deriv of potential wrt \phi
    dUdphi = np.array([m**2*phi, l**4*(twopif)*np.sin(twopif*chi)])
    #2nd deriv
    d2Udphi2 = np.array([[m**2, # V phi phi 
                          0.0],         # V phi chi
                         [0.0,          # V chi phi
                          l**4*(twopif)**2*np.cos(twopif*chi)]]) # V chi chi
    #3rd deriv Not set as not used in first order calculation
    d3Udphi3 = np.zeros((2,2,2))
    
    return U, dUdphi, d2Udphi2, d3Udphi3

def productexponential(y, params=None):
    r"""Return the potential and its first three derivatives for a product
    exponential potential.
    
    The potential is given by
    
    .. math:: 
        V = V_0 \phi^2 \exp(-\lambda \chi^2)
        
    where the parameters are :math:`V_0, \lambda`. Needs nfields=2.
    
    Parameters
    ----------
    y: array
       Array of variables with background phi as y[0]
       If you want to specify a vector of phi values, make sure
       that the first index still runs over the different 
       variables, using newaxis if necessary.
    
    params: dict
            Dictionary of parameter values labelled "lambda" , "V_0".
            
    Returns
    -------
    U, dUdphi, d2Udphi2, d3Udphi3: tuple of arrays
        Tuple of the potential and its first three derivatives.
             
    """
    
    #Check if mass is specified in params
    if params:
        l = params.get("lambda", 0.05)
        V_0 = params.get("V_0", 5.3705e-13)
    else:
        l = 0.05
        V_0 = 5.3705e-13
                
    if len(y.shape)>1:
        y = y[:,0]
             
    phi = y[0]
    chi = y[2]
    explchi2 = np.exp(-l*chi**2)
    #potential U 
    U = np.asscalar(V_0*phi**2 * explchi2)
    #deriv of potential wrt \phi
    dUdphi = np.array([2*V_0*phi*explchi2, 
                       -2*l*chi*V_0*phi**2*explchi2])
    #2nd deriv
    d2Udphi2 = np.array([[2*V_0*explchi2, # V phi phi 
                          -4*l*chi*V_0*phi*explchi2],         # V phi chi
                         [-4*l*chi*V_0*phi*explchi2,          # V chi phi
                          -2*l*V_0*phi**2*explchi2*(1-2*l*chi)]]) # V chi chi
    #3rd deriv Not set as not used in first order calculation
    d3Udphi3 = np.zeros((2,2,2))
    
    return U, dUdphi, d2Udphi2, d3Udphi3
