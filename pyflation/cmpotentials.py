# -*- coding: utf-8 -*-
"""cmpotentials.py - Cosmological potentials for cosmomodels.py

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.

    
Provides functions which can be used with cosmomodels.py. 
Default parameter values are included but can also be 
specified as a dictionary.
"""
from __future__ import division
import numpy as np

def msqphisq(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for V=1/2 m^2 phi^2
    where m is the mass of the inflaton field.
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
             
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1."""
    
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
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameter "lambda" which specifies lambda
             above.
             
    lambda can be specified in the dictionary params or otherwise
    it defaults to the value as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1."""
    #set_trace()
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
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameters "mass" and "lambda" which specifies 
             the variables.
             
    lambda can be specified in the dictionary params or otherwise
    it defaults to the value as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    
    mass can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1."""
    
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
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameters "mass" and "lambda" which specifies 
             the variables.
             
    lambda can be specified in the dictionary params or otherwise
    it defaults to the value as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1.
    
    mass can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1."""
    
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
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameter "sigma" which specifies lambda
             above.
             
    sigma can be specified in the dictionary params or otherwise
    it defaults to the value as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1."""
    #set_trace()
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
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
             
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1."""
    
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
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
             
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1."""
    
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
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
             
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1."""
    
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
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above, 
             and the parameters "c" and "d" which tune the oscillation.
             
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1."""
    
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
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
             
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1."""
    
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
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for V=1/2 m^2 phi^2 + 1/2 m^2 chi^2
    where m is the mass of the fields. Needs nfields=2.
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
             
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1."""
    
    #Check if mass is specified in params
    if params:
        m1 = params.get("m1", 1e-5)
        m2 = params.get("m2", 12e-5)
    else:
        m1 = 1e-5
        m2 = 12e-5
        
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
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameters "V0", "g", "m".
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
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for V=1/2 m^2 phi^2
    where m is the mass of the inflaton field.
    
    Arguments:
    y - Array of variables with background phi as y[0]
        If you want to specify a vector of phi values, make sure
        that the first index still runs over the different 
        variables, using newaxis if necessary.
    
    params - Dictionary of parameter values in this case should
             hold the parameter "mass" which specifies m above.
             
    m can be specified in the dictionary params or otherwise
    it defaults to the mass as normalized with the WMAP spectrum
    Pr = 2.457e-9 at the WMAP pivot scale of 0.002 Mpc^-1."""
    
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
    d2Udphi2 = mass2*np.ones((nfields,nfields), np.complex128)
    #3rd deriv
    d3Udphi3 = None
    
    return U, dUdphi, d2Udphi2, d3Udphi3