'''
Created on 29 Jul 2010

@author: ith
'''

import numpy as N

from romberg import romb

class SourceEquations(object):
    '''
    Class for source term equations
    '''


    def __init__(self, fixture):
        """Class for source term equations"""
        self.fixture = fixture
        self.fullk = N.arange(fixture["kmin"], fixture["fullkmax"], fixture["deltak"])
        self.k = self.fullk[:fixture["numsoks"]]
    
    def sourceterm(self, bgvars, a, potentials, integrand_elements, dp1, dp1dot, theta_terms):
        """Calculate the source term for this timestep"""
        pass
    
    
class SlowRollSource(SourceEquations):
    """
    Slow roll source term equations
    """
    
    def __init__(self, *args, **kwargs):
        """Class for slow roll source term equations"""
        super(SlowRollSource, self).__init__(*args, **kwargs)
        
    def J_A(self, preaterm, dp1, C1, C2):
        """Solution for J_A which is the integral for A in terms of constants C1 and C2."""
                
        q = self.k
        C1k = C1[..., N.newaxis]
        C2k = C2[..., N.newaxis]
        aterm = (C1k*q**2 + C2k*q**4) * dp1 * preaterm
        J_A = romb(aterm, self.fixture["deltak"])
        return J_A
    
    def J_B(self, prebterm, dp1, C3, C4):
        """Solution for J_B which is the integral for B in terms of constants C3 and C4."""
                
        q = self.k
        C3k = C3[..., N.newaxis]
        C4k = C4[..., N.newaxis]
        bterm = (C3k*q**3 + C4k*q**5) * dp1 * prebterm
        J_B = romb(bterm, self.fixture["deltak"])
        return J_B
    
    def J_C(self, precterm, dp1dot, C5):
        """Solution for J_C which is the integral for C in terms of constants C5."""
                
        q = self.k
        C5k = C5[..., N.newaxis]
        cterm = (C5k*q**2) * dp1dot * precterm
        J_C = romb(cterm, self.fixture["deltak"])
        return J_C
    
    def J_D(self, predterm, dp1dot, C6, C7):
        """Solution for J_D which is the integral for D in terms of constants C6 and C7."""
                
        q = self.k
        C6k = C6[..., N.newaxis]
        C7k = C7[..., N.newaxis]
        dterm = (C6k*q + C7k*q**3) * dp1dot * predterm
        J_D = romb(dterm, self.fixture["deltak"])
        return J_D
    
    def sourceterm(self, bgvars, a, potentials, integrand_elements, dp1, dp1dot, theta_terms):
        """Return unintegrated slow roll source term.
    
        The source term before integration is calculated here using the slow roll
        approximation. This function follows the revised version of Eq (5.8) in 
        Malik 06 (astro-ph/0610864v5).
        
        Parameters
        ----------
        bgvars: tuple
                Tuple of background field values in the form `(phi, phidot, H)`
        
        a: float
           Scale factor at the current timestep, `a = ainit*exp(n)`
        
        potentials: tuple
                    Tuple of potential values in the form `(U, dU, dU2, dU3)`
        
        integrand_elements: tuple 
             Contains integrand arrays in order (k, q, theta)
                
        dp1: array_like
             Array of known dp1 values
                 
        dp1dot: array_like
                Array of dpdot1 values
                 
        theta_terms: array_like
                     3-d array of integrated theta terms of shape (4, len(k), len(q))
                 
        Returns
        -------
        src_integrand: array_like
            Array containing the unintegrated source terms for all k and q modes.
            
        References
        ----------
        Malik, K. 2006, JCAP03(2007)004, astro-ph/0610864v5
        """
            #Unpack variables
        phi, phidot, H = bgvars
        k = integrand_elements[0]
        q = integrand_elements[1]
        #Calculate dphi(q) and dphi(k-q)
        dp1_q = dp1[:q.shape[-1]]
        dp1dot_q = dp1dot[q.shape[-1]]  
        #Set ones array with same shape as self.k
        onekshape = N.ones(k.shape)
        
        #Get potentials
        V, Vp, Vpp, Vppp = potentials
        
      
        #Set C_i values
        C1 = 1/H**2 * (Vppp + phidot/a**2 * (3 * a**2 * Vpp + 2 * k**2 ))
        
        C2 = 3.5 * phidot /((a*H)**2) * onekshape
        
        C3 = -4.5 / (a*H**2) * k
        
        C4 = -phidot/(a*H**2) / k
        
        C5 = -1.5 * phidot * onekshape
        
        C6 = 2 * phidot * k
        
        C7 = - phidot / k
                
        #Get component integrals
        J_A = self.J_A(theta_terms[0], dp1_q, C1, C2)
        J_B = self.J_B(theta_terms[1], dp1_q, C3, C4)
        J_C = self.J_C(theta_terms[2], dp1dot_q, C5)
        J_D = self.J_D(theta_terms[3], dp1dot_q, C6, C7)
        
        
        src = 1/((2*N.pi)**2 ) * (J_A + J_B + J_C + J_D)
        return src
    