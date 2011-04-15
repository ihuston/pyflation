''' pyflation.multipotentials - Potential functions for multifield models

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.

'''

from __future__ import division
import numpy as np

def hybridquadratic(y, params=None):
    """Return (V, dV/dphi, d2V/dphi2, d3V/dphi3) for V=1/2 m^2 phi^2 + 1/2 m^2 chi^2
    where m is the mass of the fields.
    
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
    U = 0.5*(mass2)*(y[0]**2 + y[2]**2)
    #deriv of potential wrt \phi
    dUdphi = mass2*np.array([y[0],y[2]])
    #2nd deriv
    d2Udphi2 = mass2*np.ones((2,2))
    #3rd deriv
    d3Udphi3 = 0*np.ones((2,2,2))
    
    return U, dUdphi, d2Udphi2, d3Udphi3