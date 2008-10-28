"""Second Order Cosmological Model simulations by Ian Huston
    $Id: secondorder.py,v 1.1 2008/10/28 15:34:21 ith Exp $
    
    Provides generic class CosmologicalModel that can be used as a base for explicit models."""

from cosmomodels import *
import numpy as N
from pdb import set_trace
import cmpotentials

class CanonicalFirstOrder(PhiModels):
    """First order model using efold as time variable.
       y[0] - \phi_0 : Background inflaton
       y[1] - d\phi_0/d\eta : First deriv of \phi
       y[2] - H : Hubble parameter
       y[3] - \delta\varphi_1 : First order perturbation [Real Part]
       y[4] - \delta\varphi_1^\prime : Derivative of first order perturbation [Real Part]
       y[5] - \delta\varphi_1 : First order perturbation [Imag Part]
       y[6] - \delta\varphi_1^\prime : Derivative of first order perturbation [Imag Part]
       """
    def __init__(self,  k=None, ainit=None, *args, **kwargs):
        """Initialize variables and call superclass"""
        
        super(CanonicalFirstOrder, self).__init__(*args, **kwargs)
        
        if ainit is None:
            #Don't know value of ainit yet so scale it to 1
            self.ainit = 1
        else:
            self.ainit = ainit
        
        #Let k roam for a start if not given
        if k is None:
            self.k = 10**(N.arange(10.0)-8)
        else:
            self.k = k
        
        #Initial conditions for each of the variables.
        if self.ystart is None:
            self.ystart = N.array([15.0,-0.1,0.0,1.0,0.0,1.0,0.0])   
        
        #Set initial H value if None
        if N.all(self.ystart[2] == 0.0):
            U = self.potentials(self.ystart, self.pot_params)[0]
            self.ystart[2] = self.findH(U, self.ystart)
            
        #Text for graphs
        self.plottitle = "Complex First Order Malik Model in Efold time"
        self.tname = r"$n$"
        self.ynames = [r"$\varphi_0$",
                        r"$\dot{\varphi_0}$",
                        r"$H$",
                        r"Real $\delta\varphi_1$",
                        r"Real $\dot{\delta\varphi_1}$",
                        r"Imag $\delta\varphi_1$",
                        r"Imag $\dot{\delta\varphi_1}$"]
                    
    def derivs(self, y, t):
        """Basic background equations of motion.
            dydx[0] = dy[0]/dn etc"""
        
        #get potential from function
        U, dUdphi, d2Udphi2 = self.potentials(y, self.pot_params)        
        
        #Set derivatives taking care of k type
        if type(self.k) is N.ndarray or type(self.k) is list: 
            dydx = N.zeros((7,len(self.k)))
        else:
            dydx = N.zeros(7)
            
        
        #d\phi_0/dn = y_1
        dydx[0] = y[1] 
        
        #dphi^prime/dn
        dydx[1] = -(U*y[1] + dUdphi)/(y[2]**2)
        
        #dH/dn
        dydx[2] = -0.5*(y[1]**2)*y[2]
        
        #d\deltaphi_1/dn = y[4]
        dydx[3] = y[4]
        
        #Get a
        a = self.ainit*N.exp(t)
        
        #d\deltaphi_1^prime/dn  #
        dydx[4] = (-(3 + dydx[2]/y[2])*y[4] - ((self.k/(a*y[2]))**2)*y[3]
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[3]/(y[2]**2)))
        #print dydx[4]
        
        #Complex parts
        dydx[5] = y[6]
        
        #
        dydx[6] = (-(3 + dydx[2]/y[2])*y[6]  - ((self.k/(a*y[2]))**2)*y[5]
                    -(d2Udphi2 + 2*y[1]*dUdphi + (y[1]**2)*U)*(y[5]/(y[2]**2)))
        
        return dydx