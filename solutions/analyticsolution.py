'''analyticsolution.py
Analytic solutions for the second order Klein-Gordon equation
Created on 22 Apr 2010

@author: Ian Huston
'''
from __future__ import division
import numpy as np
import scipy
from generalsolution import GeneralSolution


#Change to fortran names for compatability 
Log = scipy.log 
Sqrt = scipy.sqrt
ArcTan = scipy.arctan
Pi = scipy.pi
ArcSinh = scipy.arcsinh
ArcTanh = scipy.arctanh


class AnalyticSolution(GeneralSolution):
    """Analytic Solution base class.
    """
    
    def __init__(self, *args, **kwargs):
        """Given a fixture and a cosmomodels model instance, initialises an AnalyticSolution class instance.
        """
        super(AnalyticSolution, self).__init__(*args, **kwargs)
        self.J_terms = []
        
    def full_source_from_model(self, m, nix, **kwargs):
        """Use the data from a model at a timestep nix to calculate the full source term S."""
        #Get background values
        bgvars = m.yresult[nix, 0:3, 0]
        a = m.ainit*np.exp(m.tresult[nix])
        
        if np.any(np.isnan(bgvars[0])):
            raise AttributeError("Background values not available for this timestep.")
                       
        k = self.srceqns.k
                #Get potentials
        potentials = m.potentials(np.array([bgvars[0]]))
        Cterms = self.calculate_Cterms(bgvars, a, potentials)
        
        results = self.J_terms[0](k, Cterms, **kwargs)
        #Get component integrals
        for term in self.J_terms[1:]:
            results += term(k, Cterms, **kwargs)
                
        src = 1 / ((2*np.pi)**2) * results
        return src
    
    
class NoPhaseBunchDaviesSolution(AnalyticSolution):
    """Analytic solution using the Bunch Davies initial conditions as the first order 
    solution and with no phase information.
    
    \delta\varphi_1 = alpha/sqrt(k) 
    \dN{\delta\varphi_1} = -alpha/sqrt(k) - alpha/beta *sqrt(k)*1j 
    """
    
    def __init__(self, *args, **kwargs):
        super(NoPhaseBunchDaviesSolution, self).__init__(*args, **kwargs)
        self.J_terms = [self.J_A, self.J_B, self.J_C, self.J_D]
        
    
    def J_A(self, k, Cterms, **kwargs):
        """Solution for J_A which is the integral for A in terms of constants C1 and C2."""
        #Set limits from k
        kmin = k[0]
        kmax = k[-1]
        alpha = kwargs["alpha"]
        beta = kwargs["beta"]
        
        C1 = Cterms[0]
        C2 = Cterms[1]
        
        J_A = ((alpha ** 2 * (-(Sqrt(kmax * (-k + kmax)) * 
              (80 * C1 * (3 * k ** 2 - 14 * k * kmax + 8 * kmax ** 2) + 
                3 * C2 * (15 * k ** 4 + 10 * k ** 3 * kmax + 8 * k ** 2 * kmax ** 2 - 176 * k * kmax ** 3 + 128 * kmax ** 4))) + 
           Sqrt(kmax * (k + kmax)) * (80 * C1 * (3 * k ** 2 + 14 * k * kmax + 8 * kmax ** 2) + 
              3 * C2 * (15 * k ** 4 - 10 * k ** 3 * kmax + 8 * k ** 2 * kmax ** 2 + 176 * k * kmax ** 3 + 128 * kmax ** 4)) - 
           Sqrt((k - kmin) * kmin) * (80 * C1 * (3 * k ** 2 - 14 * k * kmin + 8 * kmin ** 2) + 
              3 * C2 * (15 * k ** 4 + 10 * k ** 3 * kmin + 8 * k ** 2 * kmin ** 2 - 176 * k * kmin ** 3 + 128 * kmin ** 4)) - 
           Sqrt(kmin) * Sqrt(k + kmin) * (80 * C1 * (3 * k ** 2 + 14 * k * kmin + 8 * kmin ** 2) + 
              3 * C2 * (15 * k ** 4 - 10 * k ** 3 * kmin + 8 * k ** 2 * kmin ** 2 + 176 * k * kmin ** 3 + 128 * kmin ** 4)) - 
           (15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * Pi) / 2. + 
           15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * ArcTan(Sqrt(kmin / (k - kmin))) + 
           15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * Log(2 * Sqrt(k)) - 
           15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * Log(2 * (Sqrt(kmax) + Sqrt(-k + kmax))) - 
           15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * Log(2 * (Sqrt(kmax) + Sqrt(k + kmax))) + 
           15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * Log(2 * (Sqrt(kmin) + Sqrt(k + kmin))))) / (2880. * k))

        return J_A
    
    def J_B(self, k, Cterms, **kwargs):
        """Solution for J_B which is the integral for B in terms of constants C3 and C4."""
        kmax = k[-1]
        kmin = k[0]
        alpha = kwargs["alpha"]
        beta = kwargs["beta"]
        
        C3 = Cterms[2]
        C4 = Cterms[3]
        
        J_B = ((alpha ** 2 * (Sqrt(kmax * (k + kmax)) * (112 * C3 * 
               (105 * k ** 4 + 250 * k ** 3 * kmax - 104 * k ** 2 * kmax ** 2 - 48 * k * kmax ** 3 + 96 * kmax ** 4) + 
              3 * C4 * (945 * k ** 6 - 630 * k ** 5 * kmax + 504 * k ** 4 * kmax ** 2 + 4688 * k ** 3 * kmax ** 3 - 2176 * k ** 2 * kmax ** 4 - 
                 1280 * k * kmax ** 5 + 2560 * kmax ** 6)) - 
           Sqrt(kmax * (-k + kmax)) * (112 * C3 * (105 * k ** 4 - 250 * k ** 3 * kmax - 104 * k ** 2 * kmax ** 2 + 48 * k * kmax ** 3 + 
                 96 * kmax ** 4) + 3 * C4 * (945 * k ** 6 + 630 * k ** 5 * kmax + 504 * k ** 4 * kmax ** 2 - 4688 * k ** 3 * kmax ** 3 - 
                 2176 * k ** 2 * kmax ** 4 + 1280 * k * kmax ** 5 + 2560 * kmax ** 6)) - 
           Sqrt(kmin) * Sqrt(k + kmin) * (112 * C3 * 
               (105 * k ** 4 + 250 * k ** 3 * kmin - 104 * k ** 2 * kmin ** 2 - 48 * k * kmin ** 3 + 96 * kmin ** 4) + 
              3 * C4 * (945 * k ** 6 - 630 * k ** 5 * kmin + 504 * k ** 4 * kmin ** 2 + 4688 * k ** 3 * kmin ** 3 - 2176 * k ** 2 * kmin ** 4 - 
                 1280 * k * kmin ** 5 + 2560 * kmin ** 6)) - 
           Sqrt((k - kmin) * kmin) * (112 * C3 * (105 * k ** 4 - 250 * k ** 3 * kmin - 104 * k ** 2 * kmin ** 2 + 48 * k * kmin ** 3 + 
                 96 * kmin ** 4) + 3 * C4 * (945 * k ** 6 + 630 * k ** 5 * kmin + 504 * k ** 4 * kmin ** 2 - 4688 * k ** 3 * kmin ** 3 - 
                 2176 * k ** 2 * kmin ** 4 + 1280 * k * kmin ** 5 + 2560 * kmin ** 6)) - 
           (105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * Pi) / 2. + 
           105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * ArcTan(Sqrt(kmin / (k - kmin))) + 
           105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * Log(2 * Sqrt(k)) - 
           105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * Log(2 * (Sqrt(kmax) + Sqrt(-k + kmax))) - 
           105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * Log(2 * (Sqrt(kmax) + Sqrt(k + kmax))) + 
           105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * Log(2 * (Sqrt(kmin) + Sqrt(k + kmin))))) / (282240. * k ** 2))

        return J_B
    
    def J_C(self, k, Cterms, **kwargs):
        """Second method for J_C"""
        kmax = k[-1]
        kmin = k[0]
        
        alpha = kwargs["alpha"]
        beta = kwargs["beta"]
        
        C5 = Cterms[4]
        
        J_C = ((alpha**2*C5*(-(Sqrt(2)*k**3*(-10000*beta**2 - (0+15360*1j)*beta*k + 6363*k**2)) - 
           Sqrt(kmax*(-k + kmax))*((0+3840*1j)*beta*(k - kmax)**2*kmax + 400*beta**2*(3*k**2 - 14*k*kmax + 8*kmax**2) + 
              9*(15*k**4 + 10*k**3*kmax - 248*k**2*kmax**2 + 336*k*kmax**3 - 128*kmax**4)) + 
           Sqrt(kmax*(k + kmax))*((0+3840*1j)*beta*kmax*(k + kmax)**2 + 400*beta**2*(3*k**2 + 14*k*kmax + 8*kmax**2) - 
              9*(-15*k**4 + 10*k**3*kmax + 248*k**2*kmax**2 + 336*k*kmax**3 + 128*kmax**4)) + 
           Sqrt((k - kmin)*kmin)*(-400*beta**2*(3*k**2 - 14*k*kmin + 8*kmin**2) - 
              (0+60*1j)*beta*(15*k**3 - 54*k**2*kmin + 8*k*kmin**2 + 16*kmin**3) + 
              9*(15*k**4 + 10*k**3*kmin - 248*k**2*kmin**2 + 336*k*kmin**3 - 128*kmin**4)) + 
           Sqrt(kmin)*Sqrt(k + kmin)*((-3840*1j)*beta*kmin*(k + kmin)**2 - 
              400*beta**2*(3*k**2 + 14*k*kmin + 8*kmin**2) + 
              9*(-15*k**4 + 10*k**3*kmin + 248*k**2*kmin**2 + 336*k*kmin**3 + 128*kmin**4)) + 
           (15*k**3*(-80*beta**2 - (0+60*1j)*beta*k + 9*k**2)*Pi)/2. + 
           15*k**3*(80*beta**2 + (0+60*1j)*beta*k - 9*k**2)*ArcTan(Sqrt(kmin/(k - kmin))) - 
           15*k**3*(80*beta**2 + 9*k**2)*Log(2*(1 + Sqrt(2))*Sqrt(k)) + 
           k**3*(Sqrt(2)*(-10000*beta**2 - (0+15360*1j)*beta*k + 6363*k**2) + 15*(80*beta**2 + 9*k**2)*Log(2*Sqrt(k)) + 
              15*(80*beta**2 + 9*k**2)*Log(2*(1 + Sqrt(2))*Sqrt(k))) - 
           15*k**3*(80*beta**2 + 9*k**2)*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
           15*k**3*(80*beta**2 + 9*k**2)*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
           15*k**3*(80*beta**2 + 9*k**2)*Log(2*(Sqrt(kmin) + Sqrt(k + kmin)))))/(14400.*beta**2*k))
        return J_C

    def J_D(self, k, Cterms, **kwargs):
        """Solution for J_D which is the integral for D in terms of constants C6 and C7."""
        kmax = k[-1]
        kmin = k[0]
        
        alpha = kwargs["alpha"]
        beta = kwargs["beta"]
        
        C6 = Cterms[5]
        C7 = Cterms[6]
        
        j1 = ((alpha ** 2 * (-240 * Sqrt((k + kmax) / kmax) * 
             (40 * C6 * (24 * k ** 3 + 9 * k ** 2 * kmax + 2 * k * kmax ** 2 - 4 * kmax ** 3) + 
               C7 * kmax * (-105 * k ** 4 - 250 * k ** 3 * kmax + 104 * k ** 2 * kmax ** 2 + 48 * k * kmax ** 3 - 
                  96 * kmax ** 4)) - 240 * Sqrt(1 - k / kmax) * 
             (40 * C6 * (24 * k ** 3 - 9 * k ** 2 * kmax + 2 * k * kmax ** 2 + 4 * kmax ** 3) + 
               C7 * kmax * (105 * k ** 4 - 250 * k ** 3 * kmax - 104 * k ** 2 * kmax ** 2 + 48 * k * kmax ** 3 + 
                  96 * kmax ** 4)) + 240 * Sqrt((k + kmin) / kmin) * 
             (40 * C6 * (24 * k ** 3 + 9 * k ** 2 * kmin + 2 * k * kmin ** 2 - 4 * kmin ** 3) + 
               C7 * kmin * (-105 * k ** 4 - 250 * k ** 3 * kmin + 104 * k ** 2 * kmin ** 2 + 48 * k * kmin ** 3 - 
                  96 * kmin ** 4)) - 240 * Sqrt(-1 + k / kmin) * 
             (40 * C6 * (24 * k ** 3 - 9 * k ** 2 * kmin + 2 * k * kmin ** 2 + 4 * kmin ** 3) + 
               C7 * kmin * (105 * k ** 4 - 250 * k ** 3 * kmin - 104 * k ** 2 * kmin ** 2 + 48 * k * kmin ** 3 + 
                  96 * kmin ** 4)) + 12600 * k ** 3 * (8 * C6 - C7 * k ** 2) * Pi - 
            25200 * k ** 3 * (8 * C6 - C7 * k ** 2) * ArcTan(Sqrt(kmin / (k - kmin))) - 
            25200 * k ** 3 * (8 * C6 - C7 * k ** 2) * Log(2 * Sqrt(k)) + 
            25200 * k ** 3 * (8 * C6 - C7 * k ** 2) * Log(2 * (Sqrt(kmax) + Sqrt(-k + kmax))) + 
            25200 * k ** 3 * (8 * C6 - C7 * k ** 2) * Log(2 * (Sqrt(kmax) + Sqrt(k + kmax))) - 
            25200 * k ** 3 * (8 * C6 - C7 * k ** 2) * Log(2 * (Sqrt(kmin) + Sqrt(k + kmin))))) / (604800. * k ** 2))
            
        j2 = ((alpha ** 2 * (-3 * kmax * Sqrt((k + kmax) / kmax) * 
             (112 * C6 * (185 * k ** 4 - 70 * k ** 3 * kmax - 168 * k ** 2 * kmax ** 2 - 16 * k * kmax ** 3 + 
                  32 * kmax ** 4) + C7 * (-945 * k ** 6 + 630 * k ** 5 * kmax + 6664 * k ** 4 * kmax ** 2 - 
                  3152 * k ** 3 * kmax ** 3 - 11136 * k ** 2 * kmax ** 4 - 1280 * k * kmax ** 5 + 2560 * kmax ** 6)) + 
            3 * Sqrt(1 - k / kmax) * kmax * (112 * C6 * 
                (185 * k ** 4 + 70 * k ** 3 * kmax - 168 * k ** 2 * kmax ** 2 + 16 * k * kmax ** 3 + 32 * kmax ** 4) + 
               C7 * (-945 * k ** 6 - 630 * k ** 5 * kmax + 6664 * k ** 4 * kmax ** 2 + 3152 * k ** 3 * kmax ** 3 - 
                  11136 * k ** 2 * kmax ** 4 + 1280 * k * kmax ** 5 + 2560 * kmax ** 6)) + 
            3 * kmin * Sqrt((k + kmin) / kmin) * 
             (112 * C6 * (185 * k ** 4 - 70 * k ** 3 * kmin - 168 * k ** 2 * kmin ** 2 - 16 * k * kmin ** 3 + 
                  32 * kmin ** 4) + C7 * (-945 * k ** 6 + 630 * k ** 5 * kmin + 6664 * k ** 4 * kmin ** 2 - 
                  3152 * k ** 3 * kmin ** 3 - 11136 * k ** 2 * kmin ** 4 - 1280 * k * kmin ** 5 + 2560 * kmin ** 6)) - 
            3 * Sqrt(-1 + k / kmin) * kmin * (112 * C6 * 
                (185 * k ** 4 + 70 * k ** 3 * kmin - 168 * k ** 2 * kmin ** 2 + 16 * k * kmin ** 3 + 32 * kmin ** 4) + 
               C7 * (-945 * k ** 6 - 630 * k ** 5 * kmin + 6664 * k ** 4 * kmin ** 2 + 3152 * k ** 3 * kmin ** 3 - 
                  11136 * k ** 2 * kmin ** 4 + 1280 * k * kmin ** 5 + 2560 * kmin ** 6)) + 
            (2835 * k ** 5 * (16 * C6 + C7 * k ** 2) * Pi) / 2. - 
            2835 * k ** 5 * (16 * C6 + C7 * k ** 2) * ArcTan(Sqrt(kmin / (k - kmin))) + 
            2835 * k ** 5 * (16 * C6 + C7 * k ** 2) * Log(2 * Sqrt(k)) - 
            2835 * k ** 5 * (16 * C6 + C7 * k ** 2) * Log(2 * (Sqrt(kmax) + Sqrt(-k + kmax))) - 
            2835 * k ** 5 * (16 * C6 + C7 * k ** 2) * Log(2 * (Sqrt(kmax) + Sqrt(k + kmax))) + 
            2835 * k ** 5 * (16 * C6 + C7 * k ** 2) * Log(2 * (Sqrt(kmin) + Sqrt(k + kmin))))) / 
        (604800. * beta ** 2 * k ** 2)) 
        
        j3 = ((alpha ** 2 * 
          (-10 * 1j * Sqrt((k + kmax) / kmax) * 
             (24 * C6 * (448 * k ** 4 - 239 * k ** 3 * kmax + 522 * k ** 2 * kmax ** 2 + 88 * k * kmax ** 3 - 
                  176 * kmax ** 4) + C7 * kmax * 
                (315 * k ** 5 - 3794 * k ** 4 * kmax - 2648 * k ** 3 * kmax ** 2 + 6000 * k ** 2 * kmax ** 3 + 
                  1408 * k * kmax ** 4 - 2816 * kmax ** 5)) + 
            10 * 1j * Sqrt(1 - k / kmax) * (24 * C6 * 
                (448 * k ** 4 + 239 * k ** 3 * kmax + 522 * k ** 2 * kmax ** 2 - 88 * k * kmax ** 3 - 176 * kmax ** 4) - 
               C7 * kmax * (315 * k ** 5 + 3794 * k ** 4 * kmax - 2648 * k ** 3 * kmax ** 2 - 6000 * k ** 2 * kmax ** 3 + 
                  1408 * k * kmax ** 4 + 2816 * kmax ** 5)) + 
            10 * 1j * Sqrt((k + kmin) / kmin) * 
             (24 * C6 * (448 * k ** 4 - 239 * k ** 3 * kmin + 522 * k ** 2 * kmin ** 2 + 88 * k * kmin ** 3 - 
                  176 * kmin ** 4) + C7 * kmin * 
                (315 * k ** 5 - 3794 * k ** 4 * kmin - 2648 * k ** 3 * kmin ** 2 + 6000 * k ** 2 * kmin ** 3 + 
                  1408 * k * kmin ** 4 - 2816 * kmin ** 5)) - 
            20 * 1j * Sqrt(-1 + k / kmin) * (384 * C6 * (k - kmin) ** 2 * (14 * k ** 2 + 5 * k * kmin + 2 * kmin ** 2) + 
               C7 * kmin * (945 * k ** 5 - 1162 * k ** 4 * kmin - 2696 * k ** 3 * kmin ** 2 + 1200 * k ** 2 * kmin ** 3 + 
                  256 * k * kmin ** 4 + 512 * kmin ** 5)) - 9450 * 1j * C7 * k ** 6 * Pi + 
            18900 * 1j * C7 * k ** 6 * ArcTan(Sqrt(kmin / (k - kmin))) + 
            3150 * 1j * k ** 3 * (72 * C6 * k + C7 * k ** 3) * Log(2 * Sqrt(k)) - 
            3150 * 1j * k ** 3 * (72 * C6 * k + C7 * k ** 3) * Log(2 * (Sqrt(kmax) + Sqrt(-k + kmax))) + 
            3150 * 1j * k ** 3 * (72 * C6 * k + C7 * k ** 3) * Log(2 * (Sqrt(kmax) + Sqrt(k + kmax))) - 
            3150 * 1j * k ** 3 * (72 * C6 * k + C7 * k ** 3) * Log(2 * (Sqrt(kmin) + Sqrt(k + kmin))))) / 
        (604800. * beta * k ** 2))
        

        return j1 + j2 + j3
    
    def calculate_Cterms(self, bgvars, a, potentials):
        """
        Calculate the constant terms needed for source integration.
        """
        k = self.srceqns.k
        phi, phidot, H = bgvars                
        #Set ones array with same shape as self.k
        onekshape = np.ones(k.shape)
        
        V, Vp, Vpp, Vppp = potentials
        
        a2 = a**2
        H2 = H**2
        aH2 = a2*H2
        k2 = k**2
        
        #Set C_i values
        C1 = 1/H2 * (Vppp + 3 * phidot * Vpp + 2 * phidot * k2 /a2 )
        
        C2 = 3.5 * phidot /(aH2) * onekshape
        
        C3 = -4.5 * phidot * k / (aH2) 
        
        C4 = -phidot / (aH2 * k)
        
        C5 = -1.5 * phidot * onekshape
        
        C6 = 2 * phidot * k
        
        C7 = - phidot / k
        
        Cterms = [C1, C2, C3, C4, C5, C6, C7]
        return Cterms
        

    def full_source_from_model(self, m, nix):
        """Use the data from a model at a timestep nix to calculate the full source term S."""
        #Get background values
        bgvars = m.yresult[nix, 0:3, 0]
        a = m.ainit*np.exp(m.tresult[nix])
        
        #Set alpha and beta
        alpha = 1/(a*np.sqrt(2))
        beta = a*bgvars[2]
        
        return super(NoPhaseBunchDaviesSolution, self).full_source_from_model(m, nix, alpha=alpha, beta=beta)
    
class SimpleInverseSolution(AnalyticSolution):
    """Analytic solution using the Bunch Davies initial conditions as the first order 
    solution and with no phase information.
    
    \delta\varphi_1 = alpha/sqrt(k) 
    \dN{\delta\varphi_1} = -alpha/sqrt(k) - alpha/beta *sqrt(k)*1j 
    """
    
    def __init__(self, *args, **kwargs):
        super(SimpleInverseSolution, self).__init__(*args, **kwargs)
        self.J_terms = [self.J_A, self.J_B, self.J_C, self.J_D]
        
    
    def J_A(self, k, Cterms, **kwargs):
        """Solution for J_A which is the integral for A in terms of constants C1 and C2."""
        #Set limits from k
        kmin = k[0]
        kmax = k[-1]
        
        C1 = Cterms[0]
        C2 = Cterms[1]
        
        J_A = 2*C1*(-0.5*k + kmax - kmin**2/(2*k)) + 2*C2*(-1/12*k**3 + kmax**2/3 - kmin**4/(4*k))
        return J_A
    
    def J_B(self, k, Cterms, **kwargs):
        """Solution for J_B which is the integral for B in terms of constants C3 and C4."""
        kmax = k[-1]
        kmin = k[0]
        
        C3 = Cterms[2] #multiplies q**3
        C4 = Cterms[3] #multiplies q**5
        
        J_B = 2/3 * C3 *(-0.75 * k**2 + kmax*k - 0.25*kmin**4/k**2 )
        J_B += 2/3 * C4 *(-1/6 * k**4 + 1/3 * kmax**3 * k - 1/6 * kmin**6/k**2)
        return J_B
    
    def J_C(self, k, Cterms, **kwargs):
        """Second method for J_C"""
        kmax = k[-1]
        kmin = k[0]
        
        C5 = Cterms[4]
        
        J_C = 2*C5*(-0.5*k + kmax - kmin**2/(2*k))         
        return J_C

    def J_D(self, k, Cterms, **kwargs):
        """Solution for J_D which is the integral for D in terms of constants C6 and C7."""
        kmax = k[-1]
        kmin = k[0]
        
        C6 = Cterms[5]
        C7 = Cterms[6]
        
        J_D = 2/3 * C6 * (1.5 - k/kmax - 0.5* (kmin/k)**2)
        J_D += 2/3 * C7 *(-0.75 * k**2 + kmax*k - 0.25*kmin**4/k**2 )
         
        return J_D
    
    def calculate_Cterms(self, bgvars, a, potentials):
        """
        Calculate the constant terms needed for source integration.
        """
        k = self.srceqns.k
        phi, phidot, H = bgvars                
        #Set ones array with same shape as self.k
        onekshape = np.ones(k.shape)
        
        V, Vp, Vpp, Vppp = potentials
        
        a2 = a**2
        H2 = H**2
        aH2 = a2*H2
        k2 = k**2
        
        #Set C_i values
        C1 = 1/H2 * (Vppp + 3 * phidot * Vpp + 2 * phidot * k2 /a2 )
        
        C2 = 3.5 * phidot /(aH2) * onekshape
        
        C3 = -4.5 * phidot * k / (aH2) 
        
        C4 = -phidot / (aH2 * k)
        
        C5 = -1.5 * phidot * onekshape
        
        C6 = 2 * phidot * k
        
        C7 = - phidot / k
        
        Cterms = [C1, C2, C3, C4, C5, C6, C7]
        return Cterms
        

    def full_source_from_model(self, m, nix):
        """Use the data from a model at a timestep nix to calculate the full source term S."""
        #Get background values
        bgvars = m.yresult[nix, 0:3, 0]
        a = m.ainit*np.exp(m.tresult[nix])
        
        #Set alpha and beta
        alpha = 1/(a*np.sqrt(2))
        beta = a*bgvars[2]
        
        return super(SimpleInverseSolution, self).full_source_from_model(m, nix, alpha=alpha, beta=beta)
    
