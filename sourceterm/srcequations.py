'''
Created on 29 Jul 2010

@author: ith
'''
from __future__ import division

import numpy as N

from romberg import romb
from sourceterm import srccython

if not "profile" in __builtins__:
    def profile(f):
        return f

def klessq(k, q, theta):
    """Return the scalar magnitude of k^i - q^i where theta is angle between vectors.
    
    Parameters
    ----------
    k: float
       Single k value to compute array for.
    
    q: array_like
       1-d array of q values to use
     
    theta: array_like
           1-d array of theta values to use
           
    Returns
    -------
    klessq: array_like
            len(q)*len(theta) array of values for
            |k^i - q^i| = \sqrt(k^2 + q^2 - 2kq cos(theta))
    """
    return k**2+q[..., N.newaxis]**2-2*k*N.outer(q,N.cos(theta))

class SourceEquations(object):
    '''
    Class for source term equations
    '''


    def __init__(self, fixture):
        """Class for source term equations"""
        self.fixture = fixture
        
        self.fullk = N.arange(fixture["kmin"], fixture["fullkmax"], fixture["deltak"])
        self.k = self.fullk[:fixture["numsoks"]]
        self.kmin = self.k[0]
        self.deltak = self.k[1] - self.k[0]
        
        self.theta = N.linspace(0, N.pi, fixture["nthetas"])
        self.dtheta = self.theta[1] - self.theta[0]
                
    
    def sourceterm(self, bgvars, a, potentials, dp1, dp1dot, theta_terms):
        """Calculate the source term for this timestep"""
        pass
    
    def getthetaterms(self, dp1, dp1dot):
        """Calculate the theta terms needed for source integrations."""
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
    
    def getthetaterms(self, dp1, dp1dot):
        """Return array of integrated values for specified theta function and dphi function.
        
        Parameters
        ----------
        dp1: array_like
             Array of values for dphi1
        
        dp1dot: array_like
                Array of values for dphi1dot
                                      
        Returns
        -------
        theta_terms: tuple
                     Tuple of len(k)xlen(q) shaped arrays of integration results in form
                     (\int(sin(theta) dp1(k-q) dtheta,
                      \int(cos(theta)sin(theta) dp1(k-q) dtheta,
                      \int(sin(theta) dp1dot(k-q) dtheta,
                      \int(cos(theta)sin(theta) dp1dot(k-q) dtheta)
                     
        """
        
        sinth = N.sin(self.theta)
        cossinth = N.cos(self.theta)*N.sin(self.theta)
        theta_terms = N.empty([4, self.k.shape[0], self.k.shape[0]])
        lenq = len(self.k)
        
        for n in xrange(len(self.k)):
            #Calculate interpolated values of dphi and dphidot
            dphi_res = srccython.interpdps2(dp1, dp1dot, self.kmin, self.deltak, n, self.theta, lenq)
            
            #Integrate theta dependence of interpolated values
            # dphi terms
            theta_terms[0,n] = romb(sinth*dphi_res[0], dx=self.dtheta)
            theta_terms[1,n] = romb(cossinth*dphi_res[0], dx=self.dtheta)
            # dphidot terms
            theta_terms[2,n] = romb(sinth*dphi_res[1], dx=self.dtheta)
            theta_terms[3,n] = romb(cossinth*dphi_res[1], dx=self.dtheta)
        return theta_terms

    def sourceterm(self, bgvars, a, potentials, dp1, dp1dot):
        """Return integrated slow roll source term.
    
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
                    
        dp1: array_like
             Array of known dp1 values
                 
        dp1dot: array_like
                Array of dpdot1 values
                 
        
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
        k = self.k

        #Calculate dphi(q) and dphi(k-q)
        dp1_q = dp1[:self.k.shape[-1]]
        dp1dot_q = dp1dot[:self.k.shape[-1]]  
        #Set ones array with same shape as self.k
        onekshape = N.ones(k.shape)
        
        theta_terms = self.getthetaterms(dp1, dp1dot)
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


class FullSingleFieldSource(SourceEquations):
    """
    Slow roll source term equations
    """
    
    def __init__(self, *args, **kwargs):
        """Class for slow roll source term equations"""
        super(FullSingleFieldSource, self).__init__(*args, **kwargs)
    
    def J_A1(self, preaterm, dp1, C1, C2):
        """Solution for J_A which is the integral for A in terms of constants C1 and C2."""
                
        q = self.k
        C1k = C1[..., N.newaxis]
        C2k = C2[..., N.newaxis]
        aterm = (C1k*q**2 + C2k*q**2*q**2) * dp1 * preaterm
        J_A = romb(aterm, self.fixture["deltak"])
        return J_A
    
    def J_A2(self, preaterm, dpdot1, C17, C18):
        """Solution for J_A2 which is the integral for A in terms of constants C17 and C18."""
                
        q = self.k
        C17k = C17[..., N.newaxis]
        C18k = C18[..., N.newaxis]
        aterm = (C17k*q**2 + C18k*q**2*q**2) * dpdot1 * preaterm
        J_A2 = romb(aterm, self.fixture["deltak"])
        return J_A2
    
    def J_B1(self, prebterm, dp1, C3, C4):
        """Solution for J_B which is the integral for B in terms of constants C3 and C4."""
                
        q = self.k
        C3k = C3[..., N.newaxis]
        C4k = C4[..., N.newaxis]
        bterm = (C3k*q**2*q + C4k*q**2*q**2*q) * dp1 * prebterm
        J_B1 = romb(bterm, self.fixture["deltak"])
        return J_B1
    
    def J_B2(self, prebterm, dpdot1, C19):
        """Solution for J_B2 which is the integral for B in terms of constant C19."""
                
        q = self.k
        C19k = C19[..., N.newaxis]
        bterm = (C19k*q**2*q) * dpdot1 * prebterm
        J_B2 = romb(bterm, self.fixture["deltak"])
        return J_B2
    
    def J_C1(self, precterm, dp1dot, C5):
        """Solution for J_C which is the integral for C in terms of constants C5."""
                
        q = self.k
        C5k = C5[..., N.newaxis]
        cterm = (C5k*q**2) * dp1dot * precterm
        J_C = romb(cterm, self.fixture["deltak"])
        return J_C
    
    def J_C2(self, precterm, dp1, C20):
        """Solution for J_C which is the integral for C in terms of constants C20."""
                
        q = self.k
        C20k = C20[..., N.newaxis]
        cterm = (C20k*q**2) * dp1 * precterm
        J_C2 = romb(cterm, self.fixture["deltak"])
        return J_C2
    
    def J_D1(self, predterm, dp1dot, C6, C7):
        """Solution for J_D which is the integral for D in terms of constants C6 and C7."""
                
        q = self.k
        C6k = C6[..., N.newaxis]
        C7k = C7[..., N.newaxis]
        dterm = (C6k*q + C7k*q**2*q) * dp1dot * predterm
        J_D = romb(dterm, self.fixture["deltak"])
        return J_D
    
    def J_D2(self, predterm, dp1, C21):
        """Solution for J_D which is the integral for D in terms of constant C21."""
                
        q = self.k
        C21k = C21[..., N.newaxis]
        dterm = (C21k*q) * dp1 * predterm
        J_D2 = romb(dterm, self.fixture["deltak"])
        return J_D2
    
    def J_E1(self, preeterm, dp1, C8, C9):
        """Solution for J_E1 which is the integral for E in terms of constants C8 and C9."""
                
        q = self.k
        C8k = C8[..., N.newaxis]
        C9k = C9[..., N.newaxis]
        eterm = (C8k*q**2 + C9k*q**2*q**2) * dp1 * preeterm
        J_E1 = romb(eterm, self.fixture["deltak"])
        return J_E1
    
    def J_E2(self, preeterm, dp1dot, C10):
        """Solution for J_E2 which is the integral for E in terms of constant C10."""
                
        q = self.k
        C10k = C10[..., N.newaxis]
        eterm = (C10k*q**2) * dp1dot * preeterm
        J_E2 = romb(eterm, self.fixture["deltak"])
        return J_E2
    
    def J_F1(self, prefterm, dp1, C11, C12):
        """Solution for J_F1 which is the integral for F in terms of constants C11 and C12."""
                
        q = self.k
        C11k = C11[..., N.newaxis]
        C12k = C12[..., N.newaxis]
        fterm = (C11k*q**2 + C12k*q**2*q**2) * dp1 * prefterm
        J_F1 = romb(fterm, self.fixture["deltak"])
        return J_F1
    
    def J_F2(self, prefterm, dpdot1, C13):
        """Solution for J_F2 which is the integral for F in terms of constant C13."""
                
        q = self.k
        C13k = C13[..., N.newaxis]
        fterm = (C13k*q**2) * dpdot1 * prefterm
        J_F2 = romb(fterm, self.fixture["deltak"])
        return J_F2
    
    def J_G1(self, pregterm, dp1, C14, C15):
        """Solution for J_G1 which is the integral for G in terms of constants C14 and C15."""
                
        q = self.k
        C14k = C14[..., N.newaxis]
        C15k = C15[..., N.newaxis]
        gterm = (C14k*q**2 + C15k*q**2*q**2) * dp1 * pregterm
        J_G1 = romb(gterm, self.fixture["deltak"])
        return J_G1
    
    def J_G2(self, pregterm, dpdot1, C16):
        """Solution for J_G2 which is the integral for G in terms of constant C16."""
                
        q = self.k
        C16k = C16[..., N.newaxis]
        gterm = (C16k*q**2) * dpdot1 * pregterm
        J_G1 = romb(gterm, self.fixture["deltak"])
        return J_G1
    
    @profile
    def getthetaterms(self, dp1, dp1dot):
        """Return array of integrated values for specified theta function and dphi function.
        
        Parameters
        ----------
        dp1: array_like
             Array of values for dphi1
        
        dp1dot: array_like
                Array of values for dphi1dot
                                      
        Returns
        -------
        theta_terms: tuple
                     Tuple of len(k)xlen(q) shaped arrays of integration results in form
                     (\int(sin(theta) dp1(k-q) dtheta,
                      \int(cos(theta)sin(theta) dp1(k-q) dtheta,
                      \int(sin(theta) dp1dot(k-q) dtheta,
                      \int(cos(theta)sin(theta) dp1dot(k-q) dtheta)
                     
        """
        
        # Sinusoidal theta terms
        sinth = N.sin(self.theta)
        cossinth = N.cos(self.theta)*sinth
        cos2sinth = N.cos(self.theta)*cossinth
        sin3th = sinth*sinth*sinth
        
        theta_terms = N.empty([7, self.k.shape[0], self.k.shape[0]])
        lenq = len(self.k)
        for n in xrange(len(self.k)):
            #klq = klessq(onek, q, theta)
            dphi_res = srccython.interpdps2(dp1, dp1dot, self.kmin, self.deltak, n, self.theta, lenq)
            
            theta_terms[0,n] = romb(sinth*dphi_res[0], dx=self.dtheta)
            theta_terms[1,n] = romb(cossinth*dphi_res[0], dx=self.dtheta)
            theta_terms[2,n] = romb(sinth*dphi_res[1], dx=self.dtheta)
            theta_terms[3,n] = romb(cossinth*dphi_res[1], dx=self.dtheta)
            
            #New terms for full solution
            # E term integration
            theta_terms[4,n] = romb(cos2sinth*dphi_res[0], dx=self.dtheta)
            #Get klessq for F and G terms
            klq2 = klessq(self.k[n], self.k, self.theta)
            sinklq = sin3th/klq2
            #Get rid of NaNs in places where dphi_res=0 or equivalently klq2<self.kmin**2
            sinklq[klq2<self.kmin**2] = 0
            # F term integration
            theta_terms[5,n] = romb(sinklq *dphi_res[0], dx=self.dtheta)
            # G term integration
            theta_terms[6,n] = romb(sinklq *dphi_res[1], dx=self.dtheta)
            
        return theta_terms

    def sourceterm(self, bgvars, a, potentials, dp1, dp1dot):
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
                
        dp1: array_like
             Array of known dp1 values
                 
        dp1dot: array_like
                Array of dpdot1 values
                 
        
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
        k = self.k
        #Calculate dphi(q) and dphi(k-q)
        dp1_q = dp1[:self.k.shape[-1]]
        dp1dot_q = dp1dot[:self.k.shape[-1]]  
        #Set ones array with same shape as self.k
        onekshape = N.ones(self.k.shape)
        
        theta_terms = self.getthetaterms(dp1, dp1dot)
        #Get potentials
        V, Vp, Vpp, Vppp = potentials
        
        a2 = a**2
        H2 = H**2
        aH2 = a2*H2
        k2 = k**2
        pdot2 = phidot**2
        
        #Calculate Q term
        Q = 1/H2 * V * phidot + a2 * Vp
      
        #Set C_i values
        C1 = (1/H2 * (Vppp + 3 * phidot * Vpp + 2 * pdot2 * Vp ) 
                + phidot/(aH2) * (2*k2 + phidot * Q + Q**2/(4*aH2)))
        
        C2 = phidot /(aH2) * (3.5 - pdot2 / 4) * onekshape
        
        C3 = 1 / (aH2) * (2 * Q * (1 - Q * phidot / (aH2)) / k - 4.5 * phidot * k)
        
        C4 = -phidot/(aH2 * k)
        
        C5 = phidot * (-1.5 + 0.25 * pdot2) * onekshape
        
        C6 = 2 * phidot * k
        
        C7 = - phidot / k
        
        C8 = phidot * Q**2 / (2*aH2**2) * onekshape
        
        C9 = phidot * pdot2 /(4*aH2) * onekshape
        
        C10 = Q * pdot2 / (2*aH2) * onekshape
        
        C11 = -phidot * Q**2 / (4*aH2) * k2
        
        C12 = -phidot * Q**2 / (2*aH2) * onekshape
        
        C13 = -pdot2 * Q / (4*aH2) * k2
        
        C14 = C13
        
        C15 = -C10
        
        C16 = -phidot * pdot2 / 4 * k2
        
        C17 = C10 * 0.5
        
        C18 =  Q * pdot2 / (aH2 * k2)
        
        C19 = -2 * Q * pdot2 / (aH2*k2)
        
        C20 = Q / (aH2) * (-2 + pdot2*(1/(2*a*H) - 0.25)) * onekshape
        
        C21 = 2 * Q / (aH2) * k 
                
        #Get component integrals
        J_A1 = self.J_A1(theta_terms[0], dp1_q, C1, C2)
        J_A2 = self.J_A2(theta_terms[0], dp1dot_q, C17, C18)
        J_B1 = self.J_B1(theta_terms[1], dp1_q, C3, C4)
        J_B2 = self.J_B2(theta_terms[1], dp1dot_q, C19)
        J_C1 = self.J_C1(theta_terms[2], dp1dot_q, C5)
        J_C2 = self.J_C2(theta_terms[2], dp1_q, C20)
        J_D1 = self.J_D1(theta_terms[3], dp1dot_q, C6, C7)
        J_D2 = self.J_D2(theta_terms[3], dp1_q, C21)
        J_E1 = self.J_E1(theta_terms[4], dp1_q, C8, C9)
        J_E2 = self.J_E2(theta_terms[4], dp1dot_q, C10)
        J_F1 = self.J_F1(theta_terms[5], dp1_q, C11, C12)
        J_F2 = self.J_F2(theta_terms[5], dp1dot_q, C13)
        J_G1 = self.J_G1(theta_terms[6], dp1_q, C14, C15)
        J_G2 = self.J_G2(theta_terms[6], dp1dot_q, C16)
        
        
        
        src = 1/((2*N.pi)**2 ) * (J_A1 + J_A2 + J_B1 + J_B2 + J_C1 + J_C2 + J_D1 + J_D2 
                                  + J_E1 + J_E2 + J_F1 + J_F2 + J_G1 + J_G2)
        return src
    
    