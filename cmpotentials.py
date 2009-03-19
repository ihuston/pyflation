"""Cosmological potentials for cosmomodels.py by Ian Huston
    $Id: cmpotentials.py,v 1.5 2009/03/19 15:16:50 ith Exp $
    
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
    Pr = 2.07e-9 at the WMAP pivot scale of 0.05 Mpc^-1."""
    
    #Check if mass is specified in params
    if params is not None and "mass" in params:
        m = params["mass"]
    else:
        #Use WMAP value of mass (in Mpl)
        m = 6.133e-6
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
    Pr = 2.07e-9 at the WMAP pivot scale of 0.05 Mpc^-1."""
    #set_trace()
    #Check if mass is specified in params
    if params is not None and "lambda" in params:
        l = params["lambda"]
    else:
        #Use WMAP value of lambda
        l = 1.5355e-13 
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