# -*- coding: utf-8 -*-
"""Cosmological potentials for cosmomodels.py by Ian Huston
    $Id: cmpotentials.py,v 1.15 2009/10/06 16:48:17 ith Exp $
    
    Provides functions which can be used with cosmomodels.py. 
    Default parameter values are included but can also be 
    specified as a dictionary."""
    
from __future__ import division
import numpy as N
from pdb import set_trace

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
    #Use inflaton mass
    mass2 = m**2
    #potential U = 1/2 m^2 \phi^2
    U = 0.5*(mass2)*(y[0]**2)
    #deriv of potential wrt \phi
    dUdphi =  (mass2)*y[0]
    #2nd deriv
    d2Udphi2 = mass2
    #3rd deriv
    d3Udphi3 = 0
    
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
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for V= lambda phi^(2/3)
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
        l = 3.81686e-10 
    if len(y.shape)>1:
        y = y[:,0]
    #potential U = 1/4 l \phi^4
    U = l*(y[0]**(2.0/3))
    #deriv of potential wrt \phi
    dUdphi =  (2.0/3)*l*(y[0]**(-1.0/3))
    #2nd deriv
    d2Udphi2 = -(2.0/9)*l*(y[0]**(-4.0/3))
    #3rd deriv
    d3Udphi3 = (8.0/27)*l*(y[0]**(-7.0/3))
    
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
        m = 6.3267e-6
    if params is not None and "V0" in params:
        V0 = params["V0"]
    else:
        V0 = 0
    #Use inflaton mass
    mass2 = m**2
    #potential U = 1/2 m^2 \phi^2
    U = 0.5*(mass2)*(y[0]**2) + V0
    #deriv of potential wrt \phi
    dUdphi =  (mass2)*y[0]
    #2nd deriv
    d2Udphi2 = mass2
    #3rd deriv
    d3Udphi3 = 0
    
    return U, dUdphi, d2Udphi2, d3Udphi3
    