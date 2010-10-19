'''
Created on 29 Jul 2010

@author: ith
'''
from __future__ import division

import numpy as np

from romberg import romb
from sourceterm import srccython

if not "profile" in __builtins__:
    def profile(f):
        return f

def klessq(k, q, theta):
    """Return the scalar magnitude of k^i - q^i squared, where theta is angle between vectors.
    
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
            |k^i - q^i|^2 = (k^2 + q^2 - 2kq cos(theta))
    """
    return k**2+q[..., np.newaxis]**2-2*k*np.outer(q,np.cos(theta))

def klessqksq(k, q, theta):
    """Return the scalar magnitude of k^i - q^i squared* 1/k**2, where theta is angle between vectors.
    
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
            1/k**2 * |k^i - q^i|^2 = (1 + (q/k)^2 - 2(q/k) cos(theta))
    """
    return 1 + (q[..., np.newaxis]/k)**2 - 2*np.outer(q/k,np.cos(theta))

class SourceEquations(object):
    '''
    Class for source term equations
    '''


    def __init__(self, fixture):
        """Class for source term equations"""
        self.fixture = fixture
        
        self.fullk = np.arange(fixture["kmin"], fixture["fullkmax"], fixture["deltak"])
        self.k = self.fullk[:fixture["numsoks"]]
        self.kmin = self.k[0]
        self.deltak = self.k[1] - self.k[0]
        
        self.theta = np.linspace(0, np.pi, fixture["nthetas"])
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
        self.J_terms = [self.J_A, self.J_B, self.J_C, self.J_D]
        
    def J_A(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_A which is the integral for A in terms of constants C1 and C2."""
                
        q = self.k
        C1k = Cterms[0][..., np.newaxis]
        C2k = Cterms[1][..., np.newaxis]
        aterm = (C1k*q**2 + C2k*q**4) * dp1 * preterms[0]
        J_A = romb(aterm, self.deltak)
        return J_A
    
    def J_B(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_B which is the integral for B in terms of constants C3 and C4."""
                
        q = self.k
        C3k = Cterms[2][..., np.newaxis]
        C4k = Cterms[3][..., np.newaxis]
        bterm = (C3k*q**3 + C4k*q**5) * dp1 * preterms[1]
        J_B = romb(bterm, self.deltak)
        return J_B
    
    def J_C(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_C which is the integral for C in terms of constants C5."""
                
        q = self.k
        C5k = Cterms[4][..., np.newaxis]
        cterm = (C5k*q**2) * dp1dot * preterms[2]
        J_C = romb(cterm, self.deltak)
        return J_C
    
    def J_D(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_D which is the integral for D in terms of constants C6 and C7."""
                
        q = self.k
        C6k = Cterms[5][..., np.newaxis]
        C7k = Cterms[6][..., np.newaxis]
        dterm = (C6k*q + C7k*q**3) * dp1dot * preterms[3]
        J_D = romb(dterm, self.deltak)
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
        
        sinth = np.sin(self.theta)
        cossinth = np.cos(self.theta)*np.sin(self.theta)
        theta_terms = np.empty([4, self.k.shape[0], self.k.shape[0]], dtype=dp1.dtype)
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

    def calculate_Cterms(self, bgvars, a, potentials):
        """
        Calculate the C terms needed for source term integration.
        """
        #Unpack variables
        phi, phidot, H = bgvars
        k = self.k
        #Set ones array with same shape as self.k
        onekshape = np.ones(k.shape)
        
        #Get potentials
        V, Vp, Vpp, Vppp = potentials
        
        a2 = a**2
        H2 = H**2
        aH2 = a2*H2
        k2 = k**2
              
        #Set C_i values
        C1 = 1/H2 * (Vppp + 3 * phidot * Vpp + 2 * phidot * k2 /a2 )
        
        C2 = 3.5 * phidot /(aH2) * onekshape
        
        C3 = -4.5 * phidot * k / (aH2)
        
        C4 = -phidot/(aH2 * k)
        
        C5 = -1.5 * phidot * onekshape
        
        C6 = 2 * phidot * k
        
        C7 = - phidot / k
        
        Cterms = [C1, C2, C3, C4, C5, C6, C7]
        return Cterms

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
            

        #Calculate dphi(q) and dphi(k-q)
        dp1_q = dp1[:self.k.shape[-1]]
        dp1dot_q = dp1dot[:self.k.shape[-1]]  
        
        
        theta_terms = self.getthetaterms(dp1, dp1dot)
        
        Cterms = self.calculate_Cterms(bgvars, a, potentials)
        
        #Get component integrals
        J_A = self.J_A(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_B = self.J_B(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_C = self.J_C(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_D = self.J_D(theta_terms, dp1_q, dp1dot_q, Cterms)
        
        
        src = 1/((2*np.pi)**2 ) * (J_A + J_B + J_C + J_D)
        return src


class FullSingleFieldSource(SourceEquations):
    """
    Full single field (non slow-roll) source term equations
    """
    
    def __init__(self, *args, **kwargs):
        """Class for slow roll source term equations"""
        super(FullSingleFieldSource, self).__init__(*args, **kwargs)
        self.J_terms = [self.J_A1, self.J_A2, self.J_B1, self.J_B2, 
                        self.J_C1, self.J_C2, self.J_D1, self.J_D2,
                        self.J_E1, self.J_E2, self.J_F1, self.J_F2,
                        self.J_G1, self.J_G2]
    
    def J_A1(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_A which is the integral for A in terms of constants C1 and C2."""
                
        q = self.k
        C1k = Cterms[0][..., np.newaxis]
        C2k = Cterms[1][..., np.newaxis]
        aterm = (C1k*q**2 + C2k*q**2*q**2) * dp1 * preterms[0]
        J_A = romb(aterm, self.deltak)
        return J_A
    
    def J_A2(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_A2 which is the integral for A in terms of constants C17 and C18."""
                
        q = self.k
        C17k = Cterms[16][..., np.newaxis]
        C18k = Cterms[17][..., np.newaxis]
        aterm = (C17k*q**2 + C18k*q**2*q**2) * dp1dot * preterms[0]
        J_A2 = romb(aterm, self.deltak)
        return J_A2
    
    def J_B1(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_B which is the integral for B in terms of constants C3 and C4."""
                
        q = self.k
        C3k = Cterms[2][..., np.newaxis]
        C4k = Cterms[3][..., np.newaxis]
        bterm = (C3k*q**2*q + C4k*q**2*q**2*q) * dp1 * preterms[1]
        J_B1 = romb(bterm, self.deltak)
        return J_B1
    
    def J_B2(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_B2 which is the integral for B in terms of constant C19."""
                
        q = self.k
        C19k = Cterms[18][..., np.newaxis]
        bterm = (C19k*q**2*q) * dp1dot * preterms[1]
        J_B2 = romb(bterm, self.deltak)
        return J_B2
    
    def J_C1(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_C which is the integral for C in terms of constants C5."""
                
        q = self.k
        C5k = Cterms[4][..., np.newaxis]
        cterm = (C5k*q**2) * dp1dot * preterms[2]
        J_C = romb(cterm, self.deltak)
        return J_C
    
    def J_C2(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_C which is the integral for C in terms of constants C20."""
                
        q = self.k
        C20k = Cterms[19][..., np.newaxis]
        cterm = (C20k*q**2) * dp1 * preterms[2]
        J_C2 = romb(cterm, self.deltak)
        return J_C2
    
    def J_D1(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_D which is the integral for D in terms of constants C6 and C7."""
                
        q = self.k
        C6k = Cterms[5][..., np.newaxis]
        C7k = Cterms[6][..., np.newaxis]
        dterm = (C6k*q + C7k*q**2*q) * dp1dot * preterms[3]
        J_D = romb(dterm, self.deltak)
        return J_D
    
    def J_D2(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_D which is the integral for D in terms of constant C21."""
                
        q = self.k
        C21k = Cterms[20][..., np.newaxis]
        dterm = (C21k*q) * dp1 * preterms[3]
        J_D2 = romb(dterm, self.deltak)
        return J_D2
    
    def J_E1(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_E1 which is the integral for E in terms of constants C8 and C9."""
                
        q = self.k
        C8k = Cterms[7][..., np.newaxis]
        C9k = Cterms[8][..., np.newaxis]
        eterm = (C8k*q**2 + C9k*q**2*q**2) * dp1 * preterms[4]
        J_E1 = romb(eterm, self.deltak)
        return J_E1
    
    def J_E2(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_E2 which is the integral for E in terms of constant C10."""
                
        q = self.k
        C10k = Cterms[9][..., np.newaxis]
        eterm = (C10k*q**2) * dp1dot * preterms[4]
        J_E2 = romb(eterm, self.deltak)
        return J_E2
    
    def J_F1(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_F1 which is the integral for F in terms of constants C11 and C12."""
                
        q = self.k
        C11k = Cterms[10][..., np.newaxis]
        C12k = Cterms[11][..., np.newaxis]
        fterm = (C11k*q**2 + C12k*q**2*q**2) * dp1 * preterms[5]
        J_F1 = romb(fterm, self.deltak)
        return J_F1
    
    def J_F2(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_F2 which is the integral for F in terms of constant C13."""
                
        q = self.k
        C13k = Cterms[12][..., np.newaxis]
        fterm = (C13k*q**2) * dp1dot * preterms[5]
        J_F2 = romb(fterm, self.deltak)
        return J_F2
    
    def J_G1(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_G1 which is the integral for G in terms of constants C14 and C15."""
                
        q = self.k
        C14k = Cterms[13][..., np.newaxis]
        C15k = Cterms[14][..., np.newaxis]
        gterm = (C14k*q**2 + C15k*q**2*q**2) * dp1 * preterms[6]
        J_G1 = romb(gterm, self.deltak)
        return J_G1
    
    def J_G2(self, preterms, dp1, dp1dot, Cterms):
        """Solution for J_G2 which is the integral for G in terms of constant C16."""
                
        q = self.k
        C16k = Cterms[15][..., np.newaxis]
        gterm = (C16k*q**2) * dp1dot * preterms[6]
        J_G1 = romb(gterm, self.deltak)
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
        sinth = np.sin(self.theta)
        cossinth = np.cos(self.theta)*sinth
        cos2sinth = np.cos(self.theta)*cossinth
        sin3th = sinth*sinth*sinth
        
        theta_terms = np.empty([7, self.k.shape[0], self.k.shape[0]], dtype=dp1.dtype)
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

    def calculate_Cterms(self, bgvars, a, potentials,):
        """
        Calculate the value of the constants needed for source term integration.
        
        """
        #Unpack variables
        phi, phidot, H = bgvars
        k = self.k
        #Get potentials
        V, Vp, Vpp, Vppp = potentials
        
        #Set ones array with same shape as self.k
        onekshape = np.ones(self.k.shape)
        
        a2 = a**2
        H2 = H**2
        aH2 = a2*H2
        k2 = k**2
        pdot2 = phidot**2
        
        #Calculate Q term
        Q = a2 * (V * phidot + Vp)
      
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
        
        C11 = -phidot * Q**2 / (4*aH2**2) * k2
        
        C12 = -phidot * Q**2 / (2*aH2**2) * onekshape
        
        C13 = -pdot2 * Q / (4*aH2) * k2
        
        C14 = C13
        
        C15 = -C10
        
        C16 = -phidot * pdot2 / 4 * k2
        
        C17 = C10 * 0.5
        
        C18 =  Q * pdot2 / (aH2 * k2)
        
        C19 = -2 * Q * pdot2 / (aH2*k)
        
        C20 = Q / (aH2) * (-2 + pdot2*(0.25)) * onekshape
        
        C21 = 2 * Q / (aH2) * k 
        
        Cterms = [C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21]
        
        return Cterms
        

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
        
        #Calculate dphi(q) and dphi(k-q)
        dp1_q = dp1[:self.k.shape[-1]]
        dp1dot_q = dp1dot[:self.k.shape[-1]]  
        
        
        theta_terms = self.getthetaterms(dp1, dp1dot)
        
        Cterms = self.calculate_Cterms(bgvars, a, potentials)
                
        #Get component integrals
        J_A1 = self.J_A1(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_A2 = self.J_A2(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_B1 = self.J_B1(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_B2 = self.J_B2(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_C1 = self.J_C1(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_C2 = self.J_C2(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_D1 = self.J_D1(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_D2 = self.J_D2(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_E1 = self.J_E1(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_E2 = self.J_E2(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_F1 = self.J_F1(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_F2 = self.J_F2(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_G1 = self.J_G1(theta_terms, dp1_q, dp1dot_q, Cterms)
        J_G2 = self.J_G2(theta_terms, dp1_q, dp1dot_q, Cterms)
        
        
        
        src = 1/((2*np.pi)**2 ) * (J_A1 + J_A2 + J_B1 + J_B2 + J_C1 + J_C2 + J_D1 + J_D2 
                                  + J_E1 + J_E2 + J_F1 + J_F2 + J_G1 + J_G2)
        return src
    
class NewFullSingleFieldSource(SourceEquations):
    """
    Full single field (non slow-roll) source term equations
    """
    
    def __init__(self, *args, **kwargs):
        """Class for slow roll source term equations"""
        super(NewFullSingleFieldSource, self).__init__(*args, **kwargs)
        self.J_params = {"A1": {"n":2, "dphiterm": "dp1", "pretermix":0},
                       "A2": {"n":3, "dphiterm": "dp1", "pretermix":0},
                       "A3": {"n":4, "dphiterm": "dp1", "pretermix":0},
                       "A4": {"n":5, "dphiterm": "dp1", "pretermix":0},
                       "A5": {"n":2, "dphiterm": "dp1dot", "pretermix":0},
                       "A6": {"n":3, "dphiterm": "dp1dot", "pretermix":0},
                       "B1": {"n":4, "dphiterm": "dp1", "pretermix":1},
                       "C1": {"n":2, "dphiterm": "dp1", "pretermix":2},
                       "C2": {"n":2, "dphiterm": "dp1dot", "pretermix":2},
                       "D1": {"n":1, "dphiterm": "dp1", "pretermix":3},
                       "D2": {"n":3, "dphiterm": "dp1", "pretermix":3},
                       "D3": {"n":1, "dphiterm": "dp1dot", "pretermix":3},
                       "D4": {"n":3, "dphiterm": "dp1dot", "pretermix":3},
                       "E1": {"n":2, "dphiterm": "dp1", "pretermix":4},
                       "E2": {"n":2, "dphiterm": "dp1dot", "pretermix":4},
                       "F1": {"n":2, "dphiterm": "dp1", "pretermix":5},
                       "F2": {"n":2, "dphiterm": "dp1dot", "pretermix":5},
                       "G1": {"n":2, "dphiterm": "dp1dot", "pretermix":6},
                       }
        

        
        self.J_terms = [self.J_factory(Jkey) for Jkey in J_params.iterkeys()]
    
    def J_factory(self, Jkey):
        def newJfunc(preterms, dp1, dp1dot, Cterms):
            return self.J_func(preterms, dp1, dp1dot, Cterms, Jkey)
        return newJfunc
    
    def J_func(self, preterms, dp1, dp1dot, Cterms, Jkey):
        """Generic solution for J_func integral."""
        q = self.k
        #Set up variables from list of Jterms and constants
        #Constant term
        Cterm = Cterms[Jkey][..., np.newaxis]
        #Index of q
        n = self.J_terms[Jkey]["n"]
        #Get text of dphiterm and set variable
        dphitermtext = self.J_terms[Jkey]["dphiterm"]
        if dphitermtext == "dp1":
            dphiterm = dp1
        elif dphitermtext == "dp1dot":
            dphiterm = dp1dot
        #Get preterm index
        pretermix = self.J_terms[Jkey]["pretermix"]
        preterm = preterms[pretermix]
        
        integrand = (Cterm*q**n) * dphiterm * preterm
        J_integral = romb(integrand, self.deltak)
        return J_integral
    
    
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
        sinth = np.sin(self.theta)
        cossinth = np.cos(self.theta)*sinth
        cos2sinth = np.cos(self.theta)*cossinth
        sin3th = sinth*sinth*sinth
        
        theta_terms = np.empty([7, self.k.shape[0], self.k.shape[0]], dtype=dp1.dtype)
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
            klq2 = klessqksq(self.k[n], self.k, self.theta)
            sinklq = sin3th/klq2
            #Get rid of NaNs in places where dphi_res=0 or equivalently klq2<self.kmin**2
            sinklq[klq2<self.kmin**2] = 0
            # F term integration
            theta_terms[5,n] = romb(sinklq *dphi_res[0], dx=self.dtheta)
            # G term integration
            theta_terms[6,n] = romb(sinklq *dphi_res[1], dx=self.dtheta)
            
        return theta_terms

    def calculate_Cterms(self, bgvars, a, potentials,):
        """
        Calculate the value of the constants needed for source term integration.
        
        """
        #Unpack variables
        phi, phidot, H = bgvars
        k = self.k
        #Get potentials
        V, Vp, Vpp, Vppp = potentials
        
        #Set ones array with same shape as self.k
        onekshape = np.ones(self.k.shape)
        
        a2 = a**2
        H2 = H**2
        aH2 = a2*H2
        pdot2 = phidot**2
        
        #Calculate Q term
        Q = a2 * (V * phidot + Vp)
        
        C_A1 = (1/H2 * (Vppp + 3*phidot*Vpp + 2*pdot2*Vp) 
                + 1/aH2 * (-0.75*pdot2*Q**2/aH2 + Q*pdot2) ) * onekshape
        
        Cterms = dict(A1 = C_A1,
                      A2 = -0.5*phidot/aH2*k,
                      A3 = 1/(aH2) * (3.5*phidot - 0.25*phidot**3) * onekshape,
                      A4 = -phidot/(k*aH2),
                      A5 = 2*Q/aH2 * onekshape,
                      A6 = -pdot2/aH2*Q/k,
                      B1 = 0.25*phidot**3/aH2 * onekshape,
                      C1 = 2*Q/aH2 * onekshape,
                      C2 = (0.25*phidot**3 - 1.5*phidot)* onekshape,
                      D1 = 2*Q*k/aH2,
                      D2 = 2*Q/(aH2*k),
                      D3 = phidot*2*k,
                      D4 = -phidot/k,
                      E1 = Q**2/(aH2**2)*phidot * onekshape,
                      E2 = pdot2/aH2 * Q * onekshape,
                      F1 = -0.25 * Q**2/(aH2**2)*phidot * onekshape,
                      F2 = -0.5*pdot2*Q/aH2 * onekshape,
                      G1 = -0.25*phidot**3 * onekshape)
        
        return Cterms
        

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
        
        #Calculate dphi(q) and dphi(k-q)
        dp1_q = dp1[:self.k.shape[-1]]
        dp1dot_q = dp1dot[:self.k.shape[-1]]  
        
        
        theta_terms = self.getthetaterms(dp1, dp1dot)
        
        Cterms = self.calculate_Cterms(bgvars, a, potentials)
            
        J_result = np.zeros(self.k.shape, dtype=dp1.dtype) 
        #Get component integrals
        for Jkey in self.J_params.iterkeys():
            J_result += self.J_func(theta_terms, dp1_q, dp1dot_q, Cterms, Jkey)
        
        src = 1/((2*np.pi)**2 ) * J_result
        return src
    
class OldSlowRollSource(SourceEquations):
    """
    Using the old slow roll equations to solve the source term.
    """
    
    def __init__(self, *args, **kwargs):
        """Class for slow roll source term equations"""
        super(OldSlowRollSource, self).__init__(*args, **kwargs)
        self.J_terms = []
        
    def calculate_Cterms(self):
        pass
        
    def getthetaterms(self,dp1, dp1dot):
        """Return array of integrated values for specified theta function and dphi function.
        
        Parameters
        ----------
        integrand_elements: tuple
                Contains integrand arrays in order (k, q, theta)
                 
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
    
        sinth = np.sin(self.theta)
        cossinth = np.cos(self.theta)*np.sin(self.theta)
        theta_terms = np.empty([4, self.k.shape[0], self.k.shape[0]], dtype=dp1.dtype)
        lenq = len(self.k)
        
        for n in xrange(len(self.k)):
            #klq = klessq(onek, q, theta)
            dphi_res = srccython.interpdps2(dp1, dp1dot, self.kmin, self.deltak, n, self.theta, lenq)
            
            theta_terms[0,n] = romb(sinth*dphi_res[0], dx=self.dtheta)
            theta_terms[1,n] = romb(cossinth*dphi_res[0], dx=self.dtheta)
            theta_terms[2,n] = romb(sinth*dphi_res[1], dx=self.dtheta)
            theta_terms[3,n] = romb(cossinth*dphi_res[1], dx=self.dtheta)
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
        U, dU, dU2, dU3 = potentials
        k = self.k[...,np.newaxis]
        q = self.k[np.newaxis, ...]
        #Calculate dphi(q) and dphi(k-q)
        dp1_q = dp1[:q.shape[-1]]
        dp1dot_q = dp1dot[:q.shape[-1]]
        aterm, bterm, cterm, dterm = self.getthetaterms(dp1, dp1dot)
        
        #Calculate unintegrated source term
        #First major term:
        if dU3!=0.0:
            src_integrand = ((1/H**2) * dU3 * q**2 * dp1_q * aterm)
        else:
            src_integrand = np.zeros_like(aterm)
        #Second major term:
        src_integrand2 = (phidot/((a*H)**2)) * ((3*dU2*(a*q)**2 + 3.5*q**4 + 2*(k**2)*(q**2))*aterm 
                          - (4.5 + (q/k)**2)* k * (q**3) * bterm) * dp1_q
        #Third major term:
        src_integrand3 = (phidot * ((-1.5*q**2)*cterm + (2 - (q/k)**2)*k*q*dterm) * dp1dot_q)
        #Multiply by prefactor
        src_integrand = src_integrand + src_integrand2 + src_integrand3
        src_integrand *= (1/(2*np.pi)**2)
        
        src = romb(src_integrand, self.deltak)
      
        return src