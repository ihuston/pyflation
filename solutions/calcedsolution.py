'''calcedsolution.py
Calculated solution for convolution integrals
Created on 22 Apr 2010

@author: Ian Huston
'''
from __future__ import division
import numpy as np
from numpy import sqrt
from generalsolution import GeneralSolution
from sosource import getthetaterms
from scipy.integrate import romb

class CalcedSolution(GeneralSolution):
    """Calculated result using romberg integration."""
    
    def __init__(self):
        super(CalcedSolution, self).__init__()
        
    def preconvolution_calced(self, fixture, alpha, beta, dp1, dp1dot):
        """Return calculates solution for pre-convolution terms."""
        #Init vars
        fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
        k = q = fullk[:fixture["numsoks"]]
         
        theta = np.linspace(0, np.pi, fixture["nthetas"])
        ie = k, q, theta
                
        tterms = getthetaterms(ie, dp1, dp1dot)
        aterm = tterms[0,0] + tterms[0,1]*1j
        bterm = tterms[1,0] + tterms[1,1]*1j
        cterm = tterms[2,0] + tterms[2,1]*1j
        dterm = tterms[3,0] + tterms[3,1]*1j
        calced_terms = [aterm, bterm, cterm, dterm]
        return calced_terms
    
    def postconvolution_calced(self, fixture, dp1, dp1dot):
        """Return calculated solution for post convolution terms."""
        preconv = self.preconvolution_calced(fixture)
        preaterm, prebterm, precterm, predterm = preconv
        
        fullk = np.arange(fixture["kmin"], fixture["fullkmax"]+fixture["deltak"], fixture["deltak"])
        q = fullk[np.newaxis, :fixture["numsoks"]]
        
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