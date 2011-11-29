'''calcedsolution.py
Calculated solution for convolution integrals

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.

'''
from __future__ import division
import numpy as np
from generalsolution import GeneralSolution

from scipy.integrate import romb

class CalcedSolution(GeneralSolution):
    """Calculated result using romberg integration."""
    
    def __init__(self, *args, **kwargs):
        super(CalcedSolution, self).__init__(*args, **kwargs)
        self.calculate_Cterms = self.srceqns.calculate_Cterms
        self.J_terms = self.srceqns.J_terms

    def get_dp1(self):
        pass
    
    def get_dp1dot(self):
        pass
    
    def full_source_from_model(self, m, nix, **kwargs):
        """Calculate full source term from model m at timestep nix."""
        
        #Get background values
        bgvars = m.yresult[nix, 0:3, 0]
        a = m.ainit*np.exp(m.tresult[nix])
                
        if np.any(np.isnan(bgvars)):
            raise AttributeError("Background values not available for this timestep.")
        
        phi = bgvars[0]
        
        dp1 = self.get_dp1(self.srceqns.fullk, **kwargs)
        dp1dot = self.get_dp1dot(self.srceqns.fullk, **kwargs)
        
        #Get potentials
        potentials = m.potentials(np.array([phi]), m.pot_params)
        
        src = self.srceqns.sourceterm(bgvars, a, potentials, dp1, dp1dot)
       
        return src
    
    
class NoPhaseBunchDaviesCalced(CalcedSolution):
    """Calced solution using the Bunch Davies initial conditions as the first order 
    solution and with no phase information.
    
    \delta\varphi_1 = alpha/sqrt(k) 
    \dN{\delta\varphi_1} = -alpha/sqrt(k) - alpha/beta *sqrt(k)*1j 
    """
        
    def __init__(self, *args, **kwargs):
        super(NoPhaseBunchDaviesCalced, self).__init__(*args, **kwargs)
        
    def get_dp1(self, k, **kwargs):
        """Get dp1 for a certain value of alpha and beta."""
        alpha = kwargs["alpha"]
        dp1 = alpha/np.sqrt(k) + 0*1j
        return dp1
    
    def get_dp1dot(self, k, **kwargs):
        """Get dp1dot for a certain value of alpha and beta."""
        alpha = kwargs["alpha"]
        beta = kwargs["beta"]
        dp1dot = -alpha/np.sqrt(k) -(alpha/beta)*np.sqrt(k) * 1j
        return dp1dot
    
        
    def full_source_from_model(self, m, nix):
        """Calculate full source term from model m at timestep nix."""
        #Get background values
        bgvars = m.yresult[nix, 0:3, 0]
        a = m.ainit*np.exp(m.tresult[nix])
        
        if np.any(np.isnan(bgvars)):
            raise AttributeError("Background values not available for this timestep.")
        #Set alpha and beta
        alpha = 1/(a*np.sqrt(2))
        beta = a*bgvars[2]
        
        return super(NoPhaseBunchDaviesCalced, self).full_source_from_model(m, nix, alpha=alpha, beta=beta)

class SimpleInverseCalced(CalcedSolution):
    """Calced solution using a simple inverse as the first order 
    solution and with no phase information.
    
    \delta\varphi_1(x) = 1/x 
    \dN{\delta\varphi_1} = 1/x 
    """
        
    def __init__(self, *args, **kwargs):
        super(SimpleInverseCalced, self).__init__(*args, **kwargs)
        
    def get_dp1(self, k, **kwargs):
        """Get dp1 for a certain value of alpha and beta."""
        dp1 = 1/k + 0*1j
        return dp1
    
    def get_dp1dot(self, k, **kwargs):
        """Get dp1dot for a certain value of alpha and beta."""
        dp1dot = 1/k + 0*1j
        return dp1dot
    
class ImaginaryInverseCalced(CalcedSolution):
    """Calced solution using an imaginary inverse as the first order 
    solution and with no phase information.
    
    \delta\varphi_1(x) = 1/x*1j 
    \dN{\delta\varphi_1} = 1/x*1j
    where j=sqrt(-1)
    """
        
    def __init__(self, *args, **kwargs):
        super(ImaginaryInverseCalced, self).__init__(*args, **kwargs)
        
    def get_dp1(self, k, **kwargs):
        """Get dp1 for a certain value of alpha and beta."""
        dp1 = 0 + (1/k)*1j
        return dp1
    
    def get_dp1dot(self, k, **kwargs):
        """Get dp1dot for a certain value of alpha and beta."""
        dp1dot = 0 + (1/k)*1j
        return dp1dot
    
