'''calcedsolution.py
Calculated solution for convolution integrals
Created on 22 Apr 2010

@author: Ian Huston
'''
from __future__ import division
import numpy as np
from generalsolution import GeneralSolution
from sosource import getthetaterms, slowrollsrcterm

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
        ie = self.k, self.k, theta
                
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
    
    def J_B(self, prebterm, dp1, C3, C4):
        """Solution for J_B which is the integral for B in terms of constants C3 and C4."""
                
        q = self.k
        bterm = (C3*q**3 + C4*q**5) * dp1 * prebterm
        J_B = romb(bterm, self.fixture["deltak"])
        return J_B
    
    def J_C(self, precterm, dp1dot, C5):
        """Solution for J_C which is the integral for C in terms of constants C5."""
                
        q = self.k
        cterm = (C5*q**2) * dp1dot * precterm
        J_C = romb(cterm, self.fixture["deltak"])
        return J_C
    
    def J_D(self, predterm, dp1dot, C6, C7):
        """Solution for J_D which is the integral for D in terms of constants C6 and C7."""
                
        q = self.k
        dterm = (C6*q + C7*q**3) * dp1dot * predterm
        J_D = romb(dterm, self.fixture["deltak"])
        return J_D
    
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
        
        phi, phidot, H = bgvars
        
        #Set alpha and beta
        alpha = 1/(a*np.sqrt(2))
        beta = a*bgvars[2]
        
        
        dp1 = self.get_dp1(self.k, alpha)
        dp1dot = self.get_dp1dot(self.k, alpha, beta)
        
        #Set ones array with same shape as self.k
        onekshape = np.ones(self.k.shape)
        
        #Get potentials
        V, Vp, Vpp, Vppp = m.potentials(np.array([phi]))
        
        #Get preterms
        preterms = self.preconvolution_calced(alpha, beta)
        
        #Set C_i values
        C1 = 1/H**2 * (Vppp + phidot/a**2 * (3 * a**2 * Vpp + 2 * self.k**2 ))
        
        C2 = 3.5 * phidot /((a*H)**2) * onekshape
        
        C3 = -4.5 / (a*H**2) * self.k
        
        C4 = -phidot/(a*H**2) / self.k
        
        C5 = -1.5 * phidot * onekshape
        
        C6 = 2 * phidot * self.k
        
        C7 = - phidot / self.k
        
        #Get component integrals
        J_A = self.J_A(self.k, preterms[0], dp1, C1, C2)
        J_B = self.J_B(self.k, preterms[1], dp1, C3, C4)
        J_C = self.J_C(self.k, preterms[2], dp1dot, C5)
        J_D = self.J_D(self.k, preterms[3], dp1dot, C6, C7)
        
        
        src = 1/((2*np.pi)**2 ) * (J_A + J_B + J_C + J_D)
        return src, (C1,C2,C3,C4,C5,C6,C7), (J_A,J_B,J_C, J_D), (alpha, beta)
        
class WithPhaseBunchDaviesCalced(CalcedSolution):
    """Calced solution using the Bunch Davies initial conditions as the first order 
    solution and with phase information.
    
    \delta\varphi_1 = alpha/sqrt(k) e^{-i*k*eta}
    \dN{\delta\varphi_1} = (-alpha/sqrt(k) - alpha/beta *sqrt(k)*1j )*e^{-iketa}
    """
        
    def __init__(self, *args, **kwargs):
        super(WithPhaseBunchDaviesCalced, self).__init__(*args, **kwargs)
        
    def get_dp1(self, k, alpha, eta):
        """Get dp1 for a certain value of alpha and beta."""
        dp1 = alpha/np.sqrt(k) * np.exp(-k*eta*1j)
        return dp1
    
    def get_dp1dot(self, k, alpha, beta, eta):
        """Get dp1dot for a certain value of alpha and beta."""
        dp1dot = (-alpha/np.sqrt(k) -(alpha/beta)*np.sqrt(k) * 1j)*np.exp(-k*eta*1j)
        return dp1dot
    
    def preconvolution_calced(self, alpha, beta, eta):
        """Return calculates solution for pre-convolution terms."""
        dp1_fullk = self.get_dp1(self.fullk, alpha, eta)
        dp1dot_fullk = self.get_dp1dot(self.fullk, alpha, beta, eta)
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
        
        #Get potentials
        potentials = m.potentials(bgvars)
        
        #Set alpha and beta
        alpha = 1/(a*np.sqrt(2))
        beta = a*bgvars[2]
        
        #Find eta using bgepsilon
        eta = -1/(beta*(1-m.bgepsilon[nix]))
        
        theta = np.linspace(0, np.pi, self.fixture["nthetas"])
        ie = self.k, self.k, theta
        
        dp1_fullk = self.get_dp1(self.fullk, alpha, eta)
        dp1dot_fullk = self.get_dp1dot(self.fullk, alpha, beta, eta)
        
        theta_terms = getthetaterms(ie, dp1_fullk, dp1dot_fullk)
        
        src_integrand = slowrollsrcterm(bgvars, a, potentials, ie, dp1_fullk, dp1dot_fullk, theta_terms)
        src = romb(src_integrand, dx=m.k[1]-m.k[0])
        return src