# -*- coding: utf-8 -*-
"""Cosmological potentials for cosmomodels.py by Ian Huston
    $Id: cmpotentials.py,v 1.18 2009/10/07 16:07:45 ith Exp $
    
    Provides functions which can be used with cosmomodels.py. 
    Default parameter values are included but can also be 
    specified as a dictionary."""
    
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
        
    yshape = np.ones_like(y[0])
    #Use inflaton mass
    mass2 = m**2
    #potential U = 1/2 m^2 \phi^2
    U = 0.5*(mass2)*(y[0]**2)
    #deriv of potential wrt \phi
    dUdphi =  (mass2)*y[0]
    #2nd deriv
    d2Udphi2 = mass2*yshape
    #3rd deriv
    d3Udphi3 = 0*yshape
    
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
    #potential U = 1/4 l \phi^4
    U = 0.25*l*(y[0]**4)
    #deriv of potential wrt \phi
    dUdphi =  l*(y[0]**3)
    #2nd deriv
    d2Udphi2 = 3*l*(y[0]**2)
    #3rd deriv
    d3Udphi3 = 6*l*(y[0])
    
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
        
    U = -0.5*(mass2)*(y[0]**2) + 0.25*l*(y[0]**4) + (m**4)/(4*l)
    #deriv of potential wrt \phi
    dUdphi =  -(mass2)*y[0] + l*(y[0]**3)
    #2nd deriv
    d2Udphi2 = -mass2 + 3*l*(y[0]**2)
    #3rd deriv
    d3Udphi3 = 6*l*(y[0])
    
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
        
    U = 0.5*(mass2)*(y[0]**2) + 0.25*l*(y[0]**4)
    #deriv of potential wrt \phi
    dUdphi =  (mass2)*y[0] + l*(y[0]**3)
    #2nd deriv
    d2Udphi2 = mass2 + 3*l*(y[0]**2)
    #3rd deriv
    d3Udphi3 = 6*l*(y[0])
    
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
    #potential U = 1/4 s \phi^4
    U = s*(y[0]**(2.0/3))
    #deriv of potential wrt \phi
    dUdphi =  (2.0/3)*s*(y[0]**(-1.0/3))
    #2nd deriv
    d2Udphi2 = -(2.0/9)*s*(y[0]**(-4.0/3))
    #3rd deriv
    d3Udphi3 = (8.0/27)*s*(y[0]**(-7.0/3))
    
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
    
    yshape = np.ones_like(y[0])
    #Use inflaton mass
    mass2 = m**2
    #potential U = 1/2 m^2 \phi^2
    U = 0.5*(mass2)*(y[0]**2) + V0
    #deriv of potential wrt \phi
    dUdphi =  (mass2)*y[0]
    #2nd deriv
    d2Udphi2 = mass2*yshape
    #3rd deriv
    d3Udphi3 = 0*yshape
    
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
    
    #Use inflaton mass
    mass2 = m**2
    #potential U = 1/2 m^2 \phi^2
    
    phisq = y[0]**2
    
    phiterm = (y[0]-phi_s)/d
    s = 1/np.cosh(phiterm)
    t = np.tanh(phiterm)
    
    U = 0.5*(mass2)*(y[0]**2) * (1 + c * t)
    #deriv of potential wrt \phi
    dUdphi =  (mass2)*y[0] * (1 + c*t) + c * mass2 * phisq * s**2 / (2*d)
    #2nd deriv
    d2Udphi2 = 0.5*mass2*(4*c*y[0]*s**2/d - 2*c*phisq*s**2*t/(d**2) + 2*(1+c*t))
    #3rd deriv
    d3Udphi3 = 0.5*mass2*(6*c*s**2/d - 12*c*y[0]*s**2*t/(d**2) 
                          + c*phisq*(-2*s**4/(d**3) + 4*s**2*t**2/(d**3)))
    
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
    
    phisq = y[0]**2
    
    phiterm = (y[0]-phi_b)/d
    s = 1/np.cosh(phiterm)
    t = np.tanh(phiterm)
    
    U = 0.5*(mass2)*(y[0]**2) * (1 + c * s)
    #deriv of potential wrt \phi
    dUdphi =  (mass2)*y[0] * (1 + c*s) - c * mass2 * phisq * s*t / (2*d)
    #2nd deriv
    d2Udphi2 = 0.5*mass2*(-4*c*y[0]*s*t/d + c*phisq*(-s**3/(d**2) + s*(t**2)/(d**2)) + 2*(1+c*s))
    #3rd deriv
    d3Udphi3 = 0.5*mass2*(-6*c*s*t/d + 6*c*y[0]*(-s**3/(d**2) + s*(t**2)/(d**2)) 
                          + c*phisq*(5*s**3*t/(d**3) - s*t**3/(d**3)))
    
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
    
    phi = y[0]
    phisq = phi*2
    
    phiterm = phi/d
    sphi = np.sin(phiterm)
    cphi = np.cos(phiterm)
    
    U = 0.5*(mass2)*(phisq) * (1 + c * sphi)
    #deriv of potential wrt \phi
    dUdphi =  (mass2)*phi * (1 + c*sphi) + c * mass2 * phisq * cphi / (2*d)
    #2nd deriv
    d2Udphi2 = mass2*((1+c*sphi) + 2*c/d * cphi * phi)
    #3rd deriv
    d3Udphi3 = mass2*(3*c/d*cphi -3*c/d**2*sphi * phi -0.5*c/d**3 *cphi * phisq)
    
    return U, dUdphi, d2Udphi2, d3Udphi3
