'''calcedsolution.py
Calculated solution for convolution integrals
Created on 22 Apr 2010

@author: Ian Huston
'''
from __future__ import division
import numpy as np
from generalsolution import GeneralSolution
from sosource import getthetaterms
from scipy.integrate import romb

class CalcedSolution(GeneralSolution):
    """Calculated result using romberg integration."""
    
    def __init__(self, *args, **kwargs):
        super(CalcedSolution, self).__init__(*args, **kwargs)
        
    def preconvolution_calced(self, dp1_fullk, dp1dot_fullk):
        """Return calculates solution for pre-convolution terms."""
        #Init vars
        fixture = self.fixture
         
        theta = np.linspace(0, np.pi, fixture["nthetas"])
        ie = self.fullk, self.fullk, theta
                
        tterms = getthetaterms(ie, dp1_fullk, dp1dot_fullk)
        aterm = tterms[0,0] + tterms[0,1]*1j
        bterm = tterms[1,0] + tterms[1,1]*1j
        cterm = tterms[2,0] + tterms[2,1]*1j
        dterm = tterms[3,0] + tterms[3,1]*1j
        calced_terms = [aterm, bterm, cterm, dterm]
        return calced_terms
    
    def postconvolution_calced(self, dp1, dp1dot):
        """Return calculated solution for post convolution terms."""
        fixture = self.fixture
        preconv = self.preconvolution_calced(fixture)
        preaterm, prebterm, precterm, predterm = preconv
        
        q = self.k
        
        aterm = 2*np.pi * q**2 * dp1 * preaterm
        integrated_a = romb(aterm, fixture["deltak"])
        
        bterm = 2*np.pi * q**2 * dp1 * prebterm
        integrated_b = romb(bterm, fixture["deltak"])
        
        cterm = 2*np.pi * q**2 * dp1 * precterm
        integrated_c = romb(cterm, fixture["deltak"])
        
        dterm = 2*np.pi * q**2 * dp1 * predterm
        integrated_d = romb(dterm, fixture["deltak"])
        
        return integrated_a, integrated_b, integrated_c, integrated_d
    
    def get_dp1(self):
        pass
    
    def get_dp1dot(self):
        pass
    
    def J_A(self, preaterm, dp1, C1, C2):
        """Solution for J_A which is the integral for A in terms of constants C1 and C2."""
                
        q = self.k
        aterm = (C1*q**2 + C2*q**4) * dp1 * preaterm
        J_A = romb(aterm, self.fixture["deltak"])
        return J_A
    
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
        dp1 = alpha/np.sqrt(k)
        return dp1
    
    def get_dp1dot(self, k, alpha, beta):
        """Get dp1dot for a certain value of alpha and beta."""
        dp1dot = -alpha/np.sqrt(k) -(alpha/beta)*np.sqrt(k) * 1j
        return dp1dot
    
    def preconvolution_calced(self, alpha, beta):
        """Return calculates solution for pre-convolution terms."""
        dp1_fullk = self.get_dp1(self.fullk, alpha)
        dp1dot_fullk = self.get_dp1dot(self.fullk, alpha, beta)
        return super(NoPhaseBunchDaviesCalced, self).preconvolution_calced(dp1_fullk, dp1dot_fullk)