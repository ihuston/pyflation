'''calcedsolution.py
Calculated solution for convolution integrals
Created on 22 Apr 2010

@author: Ian Huston
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
        
    def preconvolution_calced(self, dp1_fullk, dp1dot_fullk):
        """Return calculates solution for pre-convolution terms."""
        #Init vars
        
        tterms = self.srceqns.getthetaterms(dp1_fullk, dp1dot_fullk)
        return tterms
    
    def postconvolution_calced(self, dp1, dp1dot, dp1_fullk, dp1dot_fullk):
        """Return calculated solution for post convolution terms."""
        
        preconv = self.preconvolution_calced(dp1_fullk, dp1dot_fullk)
        preaterm, prebterm, precterm, predterm = preconv
        
        q = self.srceqns.k
        
        aterm = 2*np.pi * q**2 * dp1 * preaterm
        integrated_a = romb(aterm, self.srceqns.deltak)
        
        bterm = 2*np.pi * q**2 * dp1 * prebterm
        integrated_b = romb(bterm, self.srceqns.deltak)
        
        cterm = 2*np.pi * q**2 * dp1 * precterm
        integrated_c = romb(cterm, self.srceqns.deltak)
        
        dterm = 2*np.pi * q**2 * dp1 * predterm
        integrated_d = romb(dterm, self.srceqns.deltak)
        
        return integrated_a, integrated_b, integrated_c, integrated_d
    
    def get_dp1(self):
        pass
    
    def get_dp1dot(self):
        pass
    
class NoPhaseBunchDaviesCalced(CalcedSolution):
    """Calced solution using the Bunch Davies initial conditions as the first order 
    solution and with no phase information.
    
    \delta\varphi_1 = alpha/sqrt(k) 
    \dN{\delta\varphi_1} = -alpha/sqrt(k) - alpha/beta *sqrt(k)*1j 
    """
        
    def __init__(self, *args, **kwargs):
        super(NoPhaseBunchDaviesCalced, self).__init__(*args, **kwargs)
        
    def get_dp1(self, k, alpha):
        """Get dp1 for a certain value of alpha and beta."""
        dp1 = alpha/np.sqrt(k) + 0*1j
        return dp1
    
    def get_dp1dot(self, k, alpha, beta):
        """Get dp1dot for a certain value of alpha and beta."""
        dp1dot = -alpha/np.sqrt(k) -(alpha/beta)*np.sqrt(k) * 1j
        return dp1dot
    
    def preconvolution_calced(self, alpha, beta):
        """Return calculates solution for pre-convolution terms."""
        dp1_fullk = self.get_dp1(self.srceqns.fullk, alpha)
        dp1dot_fullk = self.get_dp1dot(self.srceqns.fullk, alpha, beta)
        return super(NoPhaseBunchDaviesCalced, self).preconvolution_calced(dp1_fullk, dp1dot_fullk)
        
    def full_source_from_model(self, m, nix):
        """Calculate full source term from model m at timestep nix."""
        try:
            #Get background values
            bgvars = m.yresult[nix, 0:3, 0]
            a = m.ainit*np.exp(m.tresult[nix])
        except AttributeError:
            raise
        
        if np.any(np.isnan(bgvars)):
            raise AttributeError("Background values not available for this timestep.")
        
        phi = bgvars[0]
        
        #Set alpha and beta
        alpha = 1/(a*np.sqrt(2))
        beta = a*bgvars[2]
        
        
        dp1 = self.get_dp1(self.srceqns.fullk, alpha)
        dp1dot = self.get_dp1dot(self.srceqns.fullk, alpha, beta)
        
        #Get potentials
        potentials = m.potentials(np.array([phi]))
        
        src = self.srceqns.sourceterm(bgvars, a, potentials, dp1, dp1dot)
       
        return src

