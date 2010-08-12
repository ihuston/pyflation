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
    
    
class NoPhaseBunchDaviesSolution(AnalyticSolution):
    """Analytic solution using the Bunch Davies initial conditions as the first order 
    solution and with no phase information.
    
    \delta\varphi_1 = alpha/sqrt(k) 
    \dN{\delta\varphi_1} = -alpha/sqrt(k) - alpha/beta *sqrt(k)*1j 
    """
    
    def __init__(self, *args, **kwargs):
        super(NoPhaseBunchDaviesSolution, self).__init__(*args, **kwargs)
        
    
    def J_A(self, k, alpha, C1, C2):
        """Solution for J_A which is the integral for A in terms of constants C1 and C2."""
        #Set limits from k
        kmin = k[0]
        kmax = k[-1]
        
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
    
    def J_B(self, k, alpha, C3, C4):
        """Solution for J_B which is the integral for B in terms of constants C3 and C4."""
        kmax = k[-1]
        kmin = k[0]
        
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
    
    def J_C(self, k, alpha, beta, C5):
        """Second method for J_C"""
        kmax = k[-1]
        kmin = k[0]
        
        j1 = ((alpha**2*C5*(-3840*1j*(k - kmax)**2*kmax*Sqrt(kmax*(-k + kmax)) + 
            3840*1j*kmax*(k + kmax)**2*Sqrt(kmax*(k + kmax)) - 
            3840*1j*kmin*(k + kmin)**2*Sqrt(kmin*(k + kmin)) - 
            60*1j*Sqrt((k - kmin)*kmin)*(15*k**3 - 54*k**2*kmin + 8*k*kmin**2 + 16*kmin**3) - 
            450*1j*k**4*Pi + 900*1j*k**4*ArcTan(1/Sqrt(-1 + k/kmin))))/(14400.*beta*k)) 
            
        j2 = ((alpha**2*C5*(-400*(k - 4*kmax)*(3*k - 2*kmax)*Sqrt(kmax*(-k + kmax)) + 
            400*Sqrt(kmax*(k + kmax))*(3*k + 2*kmax)*(k + 4*kmax) - 
            400*(k - 4*kmin)*(3*k - 2*kmin)*Sqrt((k - kmin)*kmin) - 
            400*Sqrt(kmin*(k + kmin))*(3*k + 2*kmin)*(k + 4*kmin) - 600*k**3*Pi + 
            1200*k**3*ArcTan(1/Sqrt(-1 + k/kmin)) + 1200*k**3*(ArcSinh(1) + Log(4) + Log(k)) - 
            1200*k**3*(ArcSinh(1) + Log(8*Sqrt(k)*(Sqrt(kmax) + Sqrt(-k + kmax))*
                 (Sqrt(kmax) + Sqrt(k + kmax)))) + 
            1200*k**3*Log(2*(Sqrt(kmin) + Sqrt(k + kmin)))))/(14400.*k))
             
        j3 = ((alpha**2*C5*(-9*Sqrt(kmax*(-k + kmax))*
             (15*k**4 + 10*k**3*kmax - 248*k**2*kmax**2 + 336*k*kmax**3 - 128*kmax**4) - 
            9*Sqrt(kmax*(k + kmax))*(-15*k**4 + 10*k**3*kmax + 248*k**2*kmax**2 + 
               336*k*kmax**3 + 128*kmax**4) + 
            9*Sqrt((k - kmin)*kmin)*(15*k**4 + 10*k**3*kmin - 248*k**2*kmin**2 + 
               336*k*kmin**3 - 128*kmin**4) + 
            9*Sqrt(kmin*(k + kmin))*(-15*k**4 + 10*k**3*kmin + 248*k**2*kmin**2 + 
               336*k*kmin**3 + 128*kmin**4) + (135*k**5*Pi)/2. - 
            135*k**5*ArcTan(1/Sqrt(-1 + k/kmin)) + 135*k**5*(ArcSinh(1) + Log(4) + Log(k)) - 
            135*k**5*(ArcSinh(1) + Log(8*Sqrt(k)*(Sqrt(kmax) + Sqrt(-k + kmax))*
                 (Sqrt(kmax) + Sqrt(k + kmax)))) + 
            135*k**5*Log(2*(Sqrt(kmin) + Sqrt(k + kmin)))))/(14400.*beta**2*k))
        
        return j1 + j2 + j3

    def J_D(self, k, alpha, beta, C6, C7):
        """Solution for J_D which is the integral for D in terms of constants C6 and C7."""
        kmax = k[-1]
        kmin = k[0]
        
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

    def full_source_from_model(self, m, nix):
        """Use the data from a model at a timestep nix to calculate the full source term S."""
        try:
            #Get background values
            phi, phidot, H = m.yresult[nix, 0:3, 0]
            a = m.ainit*np.exp(m.tresult[nix])
        except AttributeError:
            raise
        
        if np.any(np.isnan(phi)):
            raise AttributeError("Background values not available for this timestep.")
        
        #Get potentials
        V, Vp, Vpp, Vppp = m.potentials(np.array([phi]))
        
        #Set alpha and beta
        alpha = 1/(a*np.sqrt(2))
        beta = a*H
        
        k = self.srceqns.k
        #Set ones array with same shape as self.k
        onekshape = np.ones(k.shape)
        
        #Set C_i values
        C1 = 1/H**2 * (Vppp + phidot/a**2 * (3 * a**2 * Vpp + 2 * k**2 ))
        
        C2 = 3.5 * phidot /((a*H)**2) * onekshape
        
        C3 = -4.5 * phidot / (a*H**2) * k
        
        C4 = -phidot/(a*H**2) / k
        
        C5 = -1.5 * phidot * onekshape
        
        C6 = 2 * phidot * k
        
        C7 = - phidot / k
        
        #Get component integrals
        J_A = self.J_A(k, alpha, C1, C2)
        J_B = self.J_B(k, alpha, C3, C4)
        J_C = self.J_C(k, alpha, beta, C5)
        J_D = self.J_D(k, alpha, beta, C6, C7)
        
        src = 1 / ((2*np.pi)**2) * (J_A + J_B + J_C + J_D)
        return src
    
class ConstantPerturbationSolution(AnalyticSolution):
    """Analytic solution using a constant value for the first order perturbations 
    solution and with no phase information.
    
    \delta\varphi_1 = alpha 
    \dN{\delta\varphi_1} = -alpha - alpha/beta *1j 
    """
    
    def __init__(self, *args, **kwargs):
        super(ConstantPerturbationSolution, self).__init__(*args, **kwargs)
        
    
    def J_A(self, k, alpha, C1, C2):
        """Solution for J_A which is the integral for A in terms of constants C1 and C2."""
        #Set limits from k
        kmin = k[0]
        kmax = k[-1]
        
        J_A = ((2*alpha**2*C1*kmax**3)/3. + (2*alpha**2*C2*kmax**5)/5. - 
              (2*alpha**2*C1*kmin**3)/3. - (2*alpha**2*C2*kmin**5)/5.)
        return J_A
    
    def J_B(self, k, alpha, C3, C4):
        """Solution for J_B which is the integral for B in terms of constants C3 and C4."""
        kmax = k[-1]
        kmin = k[0]
        
        J_B = 0

        return J_B
    
    def J_C(self, k, alpha, beta, C5):
        """Second method for J_C"""
        kmax = k[-1]
        kmin = k[0]
        
        J_C = ((2*alpha**2*(1*1j + beta)**2*C5*kmax**3)/(3.*beta**2) - 
            (2*alpha**2*(1*1j + beta)**2*C5*kmin**3)/(3.*beta**2))
        
        return J_C

    def J_D(self, k, alpha, beta, C6, C7):
        """Solution for J_D which is the integral for D in terms of constants C6 and C7."""
        kmax = k[-1]
        kmin = k[0]
        
        J_D = 0

        return J_D

    def full_source_from_model(self, m, nix):
        """Use the data from a model at a timestep nix to calculate the full source term S."""
        try:
            #Get background values
            phi, phidot, H = m.yresult[nix, 0:3, 0]
            a = m.ainit*np.exp(m.tresult[nix])
        except AttributeError:
            raise
        
        if np.any(np.isnan(phi)):
            raise AttributeError("Background values not available for this timestep.")
        
        #Get potentials
        V, Vp, Vpp, Vppp = m.potentials(np.array([phi]))
        
        #Set alpha and beta
        alpha = 1/(a*np.sqrt(2))
        beta = a*H
        
        k = self.srceqns.k
        #Set ones array with same shape as self.k
        onekshape = np.ones(k.shape)
        
        #Set C_i values
        C1 = 1/H**2 * (Vppp + phidot/a**2 * (3 * a**2 * Vpp + 2 * k**2 ))
        
        C2 = 3.5 * phidot /((a*H)**2) * onekshape
        
        C3 = -4.5 * phidot / (a*H**2) * k
        
        C4 = -phidot/(a*H**2) / k
        
        C5 = -1.5 * phidot * onekshape
        
        C6 = 2 * phidot * k
        
        C7 = - phidot / k
        
        #Get component integrals
        J_A = self.J_A(k, alpha, C1, C2)
        J_B = self.J_B(k, alpha, C3, C4)
        J_C = self.J_C(k, alpha, beta, C5)
        J_D = self.J_D(k, alpha, beta, C6, C7)
        
        src = 1 / ((2*np.pi)**2) * (J_A + J_B + J_C + J_D)
        return src

class NoPhaseWithEtaSolution(AnalyticSolution):
    """Analytic solution using the full eta solution as the first order 
    solution and with no phase information.
    
    \delta\varphi_1 = alpha*(1/sqrt(k) -i/(eta*k**(-3/2))) 
    \dN{\delta\varphi_1} = -alpha/sqrt(k) - alpha/beta *sqrt(k)*1j 
    """
    
    def __init__(self, *args, **kwargs):
        super(NoPhaseWithEtaSolution, self).__init__(*args, **kwargs)
        
    
    def J_A(self, k, alpha, C1, C2, eta, kminoverride=None):
        """Solution for J_A which is the integral for A in terms of constants C1 and C2."""
        #Set limits from k
        
        kmax = k[-1]
        if kminoverride is not None:
            kmin = kminoverride
        else:
            kmin = k[0]
        
        j1 = (alpha**2*(-720*C2*k**2*Sqrt(kmax)*(Sqrt(-k + kmax) - Sqrt(k + kmax)) + 
            1920*C2*kmax**2.5*(Sqrt(-k + kmax) - Sqrt(k + kmax)) - 
            5760*C1*Sqrt(kmax)*(-Sqrt(-k + kmax) + Sqrt(k + kmax)) - 
            480*C2*k*kmax**1.5*(Sqrt(-k + kmax) + Sqrt(k + kmax)) - 
            5760*C1*Sqrt(kmin)*(Sqrt(k - kmin) - Sqrt(k + kmin)) - 
            1920*C2*kmin**2.5*(Sqrt(k - kmin) - Sqrt(k + kmin)) - 
            720*C2*k**2*Sqrt(kmin)*(-Sqrt(k - kmin) + Sqrt(k + kmin)) + 
            480*C2*k*kmin**1.5*(Sqrt(k - kmin) + Sqrt(k + kmin)) + 2880*C1*k*Pi + 360*C2*k**3*Pi - 
            5760*C1*k*ArcTan(Sqrt(kmin/(k - kmin))) - 720*C2*k**3*ArcTan(Sqrt(kmin/(k - kmin))) + 
            5760*C1*k*Log(2*Sqrt(k)) + 720*C2*k**3*Log(2*Sqrt(k)) - 
            5760*C1*k*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
            720*C2*k**3*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
            5760*C1*k*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
            720*C2*k**3*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
            5760*C1*k*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
            720*C2*k**3*Log(2*(Sqrt(kmin) + Sqrt(k + kmin)))))/(2880.*eta**2*k) 
        j2 = (alpha**2*(-240*1j*C2*k**2*kmax**1.5*(Sqrt(-k + kmax) - Sqrt(k + kmax)) + 
            1920*1j*C2*kmax**3.5*(Sqrt(-k + kmax) - Sqrt(k + kmax)) - 
            360*1j*C2*k**3*Sqrt(kmax)*(Sqrt(-k + kmax) + Sqrt(k + kmax)) - 
            960*1j*C2*k*kmax**2.5*(Sqrt(-k + kmax) + Sqrt(k + kmax)) - 
            3840*1j*C1*Sqrt(kmax)*(kmax*(-Sqrt(-k + kmax) + Sqrt(k + kmax)) + 
               k*(Sqrt(-k + kmax) + Sqrt(k + kmax))) - 
            480*1j*C2*k*kmin**2.5*(Sqrt(k - kmin) - 2*Sqrt(k + kmin)) - 
            960*1j*C2*kmin**3.5*(Sqrt(k - kmin) - 2*Sqrt(k + kmin)) + 
            120*1j*C2*k**2*kmin**1.5*(3*Sqrt(k - kmin) - 2*Sqrt(k + kmin)) + 
            180*1j*C2*k**3*Sqrt(kmin)*(3*Sqrt(k - kmin) + 2*Sqrt(k + kmin)) - 
            960*1j*C1*Sqrt(kmin)*(k*(Sqrt(k - kmin) - 4*Sqrt(k + kmin)) + 
               2*kmin*(Sqrt(k - kmin) - 2*Sqrt(k + kmin))) + 1440*1j*C1*k**2*Pi + 270*1j*C2*k**4*Pi - 
            2880*1j*C1*k**2*ArcTan(Sqrt(kmin/(k - kmin))) - 
            540*1j*C2*k**4*ArcTan(Sqrt(kmin/(k - kmin))) + 360*1j*C2*k**4*Log(2*Sqrt(k)) - 
            360*1j*C2*k**4*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
            360*1j*C2*k**4*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
            360*1j*C2*k**4*Log(2*(Sqrt(kmin) + Sqrt(k + kmin)))))/(2880.*eta*k) 
        j3 = (alpha**2*(-45*C2*k**4*Sqrt(kmax)*(Sqrt(-k + kmax) - Sqrt(k + kmax)) - 
            24*C2*k**2*kmax**2.5*(Sqrt(-k + kmax) - Sqrt(k + kmax)) - 
            384*C2*kmax**4.5*(Sqrt(-k + kmax) - Sqrt(k + kmax)) - 
            30*C2*k**3*kmax**1.5*(Sqrt(-k + kmax) + Sqrt(k + kmax)) + 
            528*C2*k*kmax**3.5*(Sqrt(-k + kmax) + Sqrt(k + kmax)) - 
            80*C1*Sqrt(kmax)*(3*k**2*(Sqrt(-k + kmax) - Sqrt(k + kmax)) + 
               8*kmax**2*(Sqrt(-k + kmax) - Sqrt(k + kmax)) - 14*k*kmax*(Sqrt(-k + kmax) + Sqrt(k + kmax)))
              - 30*C2*k**3*kmin**1.5*(Sqrt(k - kmin) - Sqrt(k + kmin)) + 
            528*C2*k*kmin**3.5*(Sqrt(k - kmin) - Sqrt(k + kmin)) - 
            45*C2*k**4*Sqrt(kmin)*(Sqrt(k - kmin) + Sqrt(k + kmin)) - 
            24*C2*k**2*kmin**2.5*(Sqrt(k - kmin) + Sqrt(k + kmin)) - 
            384*C2*kmin**4.5*(Sqrt(k - kmin) + Sqrt(k + kmin)) - 
            80*C1*Sqrt(kmin)*(-14*k*kmin*(Sqrt(k - kmin) - Sqrt(k + kmin)) + 
               3*k**2*(Sqrt(k - kmin) + Sqrt(k + kmin)) + 8*kmin**2*(Sqrt(k - kmin) + Sqrt(k + kmin))) - 
            120*C1*k**3*Pi - (45*C2*k**5*Pi)/2. + 240*C1*k**3*ArcTan(Sqrt(kmin/(k - kmin))) + 
            45*C2*k**5*ArcTan(Sqrt(kmin/(k - kmin))) + 240*C1*k**3*Log(2*Sqrt(k)) + 
            45*C2*k**5*Log(2*Sqrt(k)) - 240*C1*k**3*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
            45*C2*k**5*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
            240*C1*k**3*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
            45*C2*k**5*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
            240*C1*k**3*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
            45*C2*k**5*Log(2*(Sqrt(kmin) + Sqrt(k + kmin)))))/(2880.*k)
            
        return j1 + j2 + j3
    
    def J_B2(self, k, alpha, C3, C4, eta, kminoverride=None):
        """Solution for J_B which is the integral for B in terms of constants C3 and C4."""
        kmax = k[-1]
        if kminoverride is not None:
            kmin = kminoverride
        else:
            kmin = k[0]
            
        J_B = (alpha ** 2 * (-(Sqrt(kmax * (-k + kmax)) * 
              (280 * C3 * (420 * eta ** 2 * k ** 4 - 5 * eta * k ** 3 * (303 * 1j + 200 * eta * kmax) + 
                   k ** 2 * (11592 + 6414 * 1j * eta * kmax - 416 * eta ** 2 * kmax ** 2) + 
                   48 * kmax ** 2 * (84 + 53 * 1j * eta * kmax + 8 * eta ** 2 * kmax ** 2) + 
                   24 * k * kmax * (-126 - 87 * 1j * eta * kmax + 8 * eta ** 2 * kmax ** 2)) + 
                C4 * (28350 * eta ** 2 * k ** 6 + 525 * eta * k ** 5 * (-749*1j + 36 * eta * kmax) + 
                   70 * k ** 4 * (-7434 - 3745 * 1j * eta * kmax + 216 * eta ** 2 * kmax ** 2) + 
                   256 * kmax ** 4 * (2646 + 1855 * 1j * eta * kmax + 300 * eta ** 2 * kmax ** 2) + 
                   64 * k * kmax ** 3 * (-5733 - 5110 * 1j * eta * kmax + 600 * eta ** 2 * kmax ** 2) - 
                   48 * k ** 2 * kmax ** 2 * (-22442 - 18655 * 1j * eta * kmax + 1360 * eta ** 2 * kmax ** 2) - 
                   40 * k ** 3 * kmax * (8673 - 133 * 1j * eta * kmax + 3516 * eta ** 2 * kmax ** 2)))) + 
           Sqrt(kmax * (k + kmax)) * (280 * C3 * (420 * eta ** 2 * k ** 4 + 5 * eta * k ** 3 * (303 * 1j + 200 * eta * kmax) + 
                 k ** 2 * (11592 + 6414 * 1j * eta * kmax - 416 * eta ** 2 * kmax ** 2) + 
                 48 * kmax ** 2 * (84 + 53 * 1j * eta * kmax + 8 * eta ** 2 * kmax ** 2) - 
                 24 * k * kmax * (-126 - 87 * 1j * eta * kmax + 8 * eta ** 2 * kmax ** 2)) + 
              C4 * (28350 * eta ** 2 * k ** 6 - 525 * eta * k ** 5 * (-749*1j + 36 * eta * kmax) + 
                 70 * k ** 4 * (-7434 - 3745 * 1j * eta * kmax + 216 * eta ** 2 * kmax ** 2) + 
                 256 * kmax ** 4 * (2646 + 1855 * 1j * eta * kmax + 300 * eta ** 2 * kmax ** 2) - 
                 64 * k * kmax ** 3 * (-5733 - 5110 * 1j * eta * kmax + 600 * eta ** 2 * kmax ** 2) - 
                 48 * k ** 2 * kmax ** 2 * (-22442 - 18655 * 1j * eta * kmax + 1360 * eta ** 2 * kmax ** 2) + 
                 40 * k ** 3 * kmax * (8673 - 133 * 1j * eta * kmax + 3516 * eta ** 2 * kmax ** 2))) - 
           Sqrt(kmin) * Sqrt(k + kmin) * (280 * C3 * 
               (420 * eta ** 2 * k ** 4 + 5 * eta * k ** 3 * (303 * 1j + 200 * eta * kmin) + 
                k ** 2 * (11592 + 6414 * 1j * eta * kmin - 416 * eta ** 2 * kmin ** 2) + 
                 48 * kmin ** 2 * (84 + 53 * 1j * eta * kmin + 8 * eta ** 2 * kmin ** 2) - 
                24 * k * kmin * (-126 - 87 * 1j * eta * kmin + 8 * eta ** 2 * kmin ** 2)) + 
              C4 * (28350 * eta ** 2 * k ** 6 - 525 * eta * k ** 5 * (-749*1j + 36 * eta * kmin) + 
                 70 * k ** 4 * (-7434 - 3745 * 1j * eta * kmin + 216 * eta ** 2 * kmin ** 2) + 
                 256 * kmin ** 4 * (2646 + 1855 * 1j * eta * kmin + 300 * eta ** 2 * kmin ** 2) - 
                 64 * k * kmin ** 3 * (-5733 - 5110 * 1j * eta * kmin + 600 * eta ** 2 * kmin ** 2) - 
                 48 * k ** 2 * kmin ** 2 * (-22442 - 18655 * 1j * eta * kmin + 1360 * eta ** 2 * kmin ** 2) + 
                 40 * k ** 3 * kmin * (8673 - 133 * 1j * eta * kmin + 3516 * eta ** 2 * kmin ** 2))) - 
           Sqrt((k - kmin) * kmin) * (280 * C3 * (420 * eta ** 2 * k ** 4 + 5 * eta * k ** 3 * (1083 * 1j - 200 * eta * kmin) + 
                 24 * k * kmin * (126 + 67 * 1j * eta * kmin + 8 * eta ** 2 * kmin ** 2) + 
                 48 * kmin ** 2 * (-84 - 73 * 1j * eta * kmin + 8 * eta ** 2 * kmin ** 2) - 
                 2 * k ** 2 * (5796 + 2547 * 1j * eta * kmin + 208 * eta ** 2 * kmin ** 2)) + 
              C4 * (28350 * eta ** 2 * k ** 6 + 525 * eta * k ** 5 * (553 * 1j + 36 * eta * kmin) + 
                 70 * k ** 4 * (7434 + 2765 * 1j * eta * kmin + 216 * eta ** 2 * kmin ** 2) + 
                 256 * kmin ** 4 * (-2646 - 2555 * 1j * eta * kmin + 300 * eta ** 2 * kmin ** 2) + 
                 64 * k * kmin ** 3 * (5733 + 3710 * 1j * eta * kmin + 600 * eta ** 2 * kmin ** 2) - 
                 48 * k ** 2 * kmin ** 2 * (22442 + 15155 * 1j * eta * kmin + 1360 * eta ** 2 * kmin ** 2) - 
                 40 * k ** 3 * kmin * (-8673 - 9247 * 1j * eta * kmin + 3516 * eta ** 2 * kmin ** 2))) - 
           (105 * k ** 3 * (280 * C3 * (120 + 15 * 1j * eta * k + 4 * eta ** 2 * k ** 2) + 
                C4 * k ** 2 * (4956 + 2765 * 1j * eta * k + 270 * eta ** 2 * k ** 2)) * Pi) / 2. + 
           105 * k ** 3 * (280 * C3 * (120 + 15 * 1j * eta * k + 4 * eta ** 2 * k ** 2) + 
              C4 * k ** 2 * (4956 + 2765 * 1j * eta * k + 270 * eta ** 2 * k ** 2)) * ArcTan(Sqrt(kmin / (k - kmin))) + 
           105 * k ** 3 * (280 * C3 * (-120 - 51 * 1j * eta * k + 4 * eta ** 2 * k ** 2) + 
              C4 * k ** 2 * (-4956 - 3745 * 1j * eta * k + 270 * eta ** 2 * k ** 2)) * Log(2 * Sqrt(k)) - 
           105 * k ** 3 * (280 * C3 * (-120 - 51 * 1j * eta * k + 4 * eta ** 2 * k ** 2) + 
              C4 * k ** 2 * (-4956 - 3745 * 1j * eta * k + 270 * eta ** 2 * k ** 2)) * Log(2 * (Sqrt(kmax) + Sqrt(-k + kmax))) - 
           105 * k ** 3 * (280 * C3 * (-120 + 51 * 1j * eta * k + 4 * eta ** 2 * k ** 2) + 
              C4 * k ** 2 * (-4956 + 3745 * 1j * eta * k + 270 * eta ** 2 * k ** 2)) * Log(2 * (Sqrt(kmax) + Sqrt(k + kmax))) + 
           105 * k ** 3 * (280 * C3 * (-120 + 51 * 1j * eta * k + 4 * eta ** 2 * k ** 2) + 
              C4 * k ** 2 * (-4956 + 3745 * 1j * eta * k + 270 * eta ** 2 * k ** 2)) * Log(2 * (Sqrt(kmin) + Sqrt(k + kmin))))) / (2.8224e6 * eta ** 2 * k ** 2)
              
        return J_B

    def J_B(self, k, alpha, C3, C4, eta, kminoverride=None):
        """Solution for J_B which is the integral for B in terms of constants C3 and C4."""
        kmax = k[-1]
        if kminoverride is not None:
            kmin = kminoverride
        else:
            kmin = k[0]
            
        J_B = ((-4*alpha**2*C4*kmin**6*Sqrt((k - kmin)*kmin))/(147.*k**2) - 
        (4*alpha**2*C4*kmin**6.5*Sqrt(k + kmin))/(147.*k**2) + 
        (alpha**2*kmin**5*(654080*1j*C4*eta*Sqrt((k - kmin)*kmin) - 
             38400*C4*eta**2*k*Sqrt((k - kmin)*kmin)))/(2.8224e6*eta**2*k**2) + 
        (alpha**2*kmin**4*(677376*C4*Sqrt((k - kmin)*kmin) - 107520*C3*eta**2*Sqrt((k - kmin)*kmin) - 
             237440*1j*C4*eta*k*Sqrt((k - kmin)*kmin) + 65280*C4*eta**2*k**2*Sqrt((k - kmin)*kmin)))/
         (2.8224e6*eta**2*k**2) + (alpha**2*kmin**3*
           (981120*1j*C3*eta*Sqrt((k - kmin)*kmin) - 366912*C4*k*Sqrt((k - kmin)*kmin) - 
             53760*C3*eta**2*k*Sqrt((k - kmin)*kmin) + 727440*1j*C4*eta*k**2*Sqrt((k - kmin)*kmin) + 
             140640*C4*eta**2*k**3*Sqrt((k - kmin)*kmin)))/(2.8224e6*eta**2*k**2) + 
        (alpha**2*kmin**2*(1128960*C3*Sqrt((k - kmin)*kmin) - 450240*1j*C3*eta*k*Sqrt((k - kmin)*kmin) + 
             1077216*C4*k**2*Sqrt((k - kmin)*kmin) + 116480*C3*eta**2*k**2*Sqrt((k - kmin)*kmin) - 
             369880*1j*C4*eta*k**3*Sqrt((k - kmin)*kmin) - 15120*C4*eta**2*k**4*Sqrt((k - kmin)*kmin)))/
         (2.8224e6*eta**2*k**2) + (alpha**2*kmin*
           (-846720*C3*k*Sqrt((k - kmin)*kmin) + 1426320*1j*C3*eta*k**2*Sqrt((k - kmin)*kmin) - 
             346920*C4*k**3*Sqrt((k - kmin)*kmin) + 280000*C3*eta**2*k**3*Sqrt((k - kmin)*kmin) - 
             193550*1j*C4*eta*k**4*Sqrt((k - kmin)*kmin) - 18900*C4*eta**2*k**5*Sqrt((k - kmin)*kmin)))/
         (2.8224e6*eta**2*k**2) + (alpha**2*kmin**5.5*
           (-474880*1j*C4*eta*Sqrt(k + kmin) + 38400*C4*eta**2*k*Sqrt(k + kmin)))/(2.8224e6*eta**2*k**2) + 
        (alpha**2*kmin**4.5*(-677376*C4*Sqrt(k + kmin) - 107520*C3*eta**2*Sqrt(k + kmin) - 
             327040*1j*C4*eta*k*Sqrt(k + kmin) + 65280*C4*eta**2*k**2*Sqrt(k + kmin)))/
         (2.8224e6*eta**2*k**2) + (alpha**2*kmin**3.5*
           (-712320*1j*C3*eta*Sqrt(k + kmin) - 366912*C4*k*Sqrt(k + kmin) + 
             53760*C3*eta**2*k*Sqrt(k + kmin) - 895440*1j*C4*eta*k**2*Sqrt(k + kmin) - 
             140640*C4*eta**2*k**3*Sqrt(k + kmin)))/(2.8224e6*eta**2*k**2) + 
        (alpha**2*kmin**2.5*(-1128960*C3*Sqrt(k + kmin) - 584640*1j*C3*eta*k*Sqrt(k + kmin) - 
             1077216*C4*k**2*Sqrt(k + kmin) + 116480*C3*eta**2*k**2*Sqrt(k + kmin) + 
             5320*1j*C4*eta*k**3*Sqrt(k + kmin) - 15120*C4*eta**2*k**4*Sqrt(k + kmin)))/
         (2.8224e6*eta**2*k**2) + (alpha**2*kmin**1.5*
           (-846720*C3*k*Sqrt(k + kmin) - 1795920*1j*C3*eta*k**2*Sqrt(k + kmin) - 
             346920*C4*k**3*Sqrt(k + kmin) - 280000*C3*eta**2*k**3*Sqrt(k + kmin) + 
             262150*1j*C4*eta*k**4*Sqrt(k + kmin) + 18900*C4*eta**2*k**5*Sqrt(k + kmin)))/
         (2.8224e6*eta**2*k**2) + (alpha**2*Sqrt(kmin)*
           (-3245760*C3*k**2*Sqrt(k + kmin) - 424200*1j*C3*eta*k**3*Sqrt(k + kmin) + 
             520380*C4*k**4*Sqrt(k + kmin) - 117600*C3*eta**2*k**4*Sqrt(k + kmin) - 
             393225*1j*C4*eta*k**5*Sqrt(k + kmin) - 28350*C4*eta**2*k**6*Sqrt(k + kmin)))/
         (2.8224e6*eta**2*k**2) + (alpha**2*
           (-3245760*C3*k**2*Sqrt(kmax*(-k + kmax)) + 424200*1j*C3*eta*k**3*Sqrt(kmax*(-k + kmax)) + 
             520380*C4*k**4*Sqrt(kmax*(-k + kmax)) - 117600*C3*eta**2*k**4*Sqrt(kmax*(-k + kmax)) + 
             393225*1j*C4*eta*k**5*Sqrt(kmax*(-k + kmax)) - 28350*C4*eta**2*k**6*Sqrt(kmax*(-k + kmax)) + 
             846720*C3*k*kmax*Sqrt(kmax*(-k + kmax)) - 
             1795920*1j*C3*eta*k**2*kmax*Sqrt(kmax*(-k + kmax)) + 
             346920*C4*k**3*kmax*Sqrt(kmax*(-k + kmax)) + 
             280000*C3*eta**2*k**3*kmax*Sqrt(kmax*(-k + kmax)) + 
             262150*1j*C4*eta*k**4*kmax*Sqrt(kmax*(-k + kmax)) - 
             18900*C4*eta**2*k**5*kmax*Sqrt(kmax*(-k + kmax)) - 1128960*C3*kmax**2*Sqrt(kmax*(-k + kmax)) + 
             584640*1j*C3*eta*k*kmax**2*Sqrt(kmax*(-k + kmax)) - 
             1077216*C4*k**2*kmax**2*Sqrt(kmax*(-k + kmax)) + 
             116480*C3*eta**2*k**2*kmax**2*Sqrt(kmax*(-k + kmax)) - 
             5320*1j*C4*eta*k**3*kmax**2*Sqrt(kmax*(-k + kmax)) - 
             15120*C4*eta**2*k**4*kmax**2*Sqrt(kmax*(-k + kmax)) - 
             712320*1j*C3*eta*kmax**3*Sqrt(kmax*(-k + kmax)) + 
             366912*C4*k*kmax**3*Sqrt(kmax*(-k + kmax)) - 
             53760*C3*eta**2*k*kmax**3*Sqrt(kmax*(-k + kmax)) - 
             895440*1j*C4*eta*k**2*kmax**3*Sqrt(kmax*(-k + kmax)) + 
             140640*C4*eta**2*k**3*kmax**3*Sqrt(kmax*(-k + kmax)) - 
             677376*C4*kmax**4*Sqrt(kmax*(-k + kmax)) - 107520*C3*eta**2*kmax**4*Sqrt(kmax*(-k + kmax)) + 
             327040*1j*C4*eta*k*kmax**4*Sqrt(kmax*(-k + kmax)) + 
             65280*C4*eta**2*k**2*kmax**4*Sqrt(kmax*(-k + kmax)) - 
             474880*1j*C4*eta*kmax**5*Sqrt(kmax*(-k + kmax)) - 
             38400*C4*eta**2*k*kmax**5*Sqrt(kmax*(-k + kmax)) - 
             76800*C4*eta**2*kmax**6*Sqrt(kmax*(-k + kmax)) + 3245760*C3*k**2*Sqrt(kmax*(k + kmax)) + 
             424200*1j*C3*eta*k**3*Sqrt(kmax*(k + kmax)) - 520380*C4*k**4*Sqrt(kmax*(k + kmax)) + 
             117600*C3*eta**2*k**4*Sqrt(kmax*(k + kmax)) + 393225*1j*C4*eta*k**5*Sqrt(kmax*(k + kmax)) + 
             28350*C4*eta**2*k**6*Sqrt(kmax*(k + kmax)) + 846720*C3*k*kmax*Sqrt(kmax*(k + kmax)) + 
             1795920*1j*C3*eta*k**2*kmax*Sqrt(kmax*(k + kmax)) + 
             346920*C4*k**3*kmax*Sqrt(kmax*(k + kmax)) + 280000*C3*eta**2*k**3*kmax*Sqrt(kmax*(k + kmax)) - 
             262150*1j*C4*eta*k**4*kmax*Sqrt(kmax*(k + kmax)) - 
             18900*C4*eta**2*k**5*kmax*Sqrt(kmax*(k + kmax)) + 1128960*C3*kmax**2*Sqrt(kmax*(k + kmax)) + 
             584640*1j*C3*eta*k*kmax**2*Sqrt(kmax*(k + kmax)) + 
             1077216*C4*k**2*kmax**2*Sqrt(kmax*(k + kmax)) - 
             116480*C3*eta**2*k**2*kmax**2*Sqrt(kmax*(k + kmax)) - 
             5320*1j*C4*eta*k**3*kmax**2*Sqrt(kmax*(k + kmax)) + 
             15120*C4*eta**2*k**4*kmax**2*Sqrt(kmax*(k + kmax)) + 
             712320*1j*C3*eta*kmax**3*Sqrt(kmax*(k + kmax)) + 366912*C4*k*kmax**3*Sqrt(kmax*(k + kmax)) - 
             53760*C3*eta**2*k*kmax**3*Sqrt(kmax*(k + kmax)) + 
             895440*1j*C4*eta*k**2*kmax**3*Sqrt(kmax*(k + kmax)) + 
             140640*C4*eta**2*k**3*kmax**3*Sqrt(kmax*(k + kmax)) + 
             677376*C4*kmax**4*Sqrt(kmax*(k + kmax)) + 107520*C3*eta**2*kmax**4*Sqrt(kmax*(k + kmax)) + 
             327040*1j*C4*eta*k*kmax**4*Sqrt(kmax*(k + kmax)) - 
             65280*C4*eta**2*k**2*kmax**4*Sqrt(kmax*(k + kmax)) + 
             474880*1j*C4*eta*kmax**5*Sqrt(kmax*(k + kmax)) - 
             38400*C4*eta**2*k*kmax**5*Sqrt(kmax*(k + kmax)) + 
             76800*C4*eta**2*kmax**6*Sqrt(kmax*(k + kmax)) + 3245760*C3*k**2*Sqrt((k - kmin)*kmin) - 
             1516200*1j*C3*eta*k**3*Sqrt((k - kmin)*kmin) - 520380*C4*k**4*Sqrt((k - kmin)*kmin) - 
             117600*C3*eta**2*k**4*Sqrt((k - kmin)*kmin) - 290325*1j*C4*eta*k**5*Sqrt((k - kmin)*kmin) - 
             28350*C4*eta**2*k**6*Sqrt((k - kmin)*kmin) - 1764000*C3*k**3*Pi - 220500*1j*C3*eta*k**4*Pi - 
             260190*C4*k**5*Pi - 58800*C3*eta**2*k**5*Pi - 145162.5*1j*C4*eta*k**6*Pi - 
             14175*C4*eta**2*k**7*Pi + 3528000*C3*k**3*ArcTan(Sqrt(kmin/(k - kmin))) + 
             441000*1j*C3*eta*k**4*ArcTan(Sqrt(kmin/(k - kmin))) + 
             520380*C4*k**5*ArcTan(Sqrt(kmin/(k - kmin))) + 
             117600*C3*eta**2*k**5*ArcTan(Sqrt(kmin/(k - kmin))) + 
             290325*1j*C4*eta*k**6*ArcTan(Sqrt(kmin/(k - kmin))) + 
             28350*C4*eta**2*k**7*ArcTan(Sqrt(kmin/(k - kmin))) - 3528000*C3*k**3*Log(2*Sqrt(k)) - 
             1499400*1j*C3*eta*k**4*Log(2*Sqrt(k)) - 520380*C4*k**5*Log(2*Sqrt(k)) + 
             117600*C3*eta**2*k**5*Log(2*Sqrt(k)) - 393225*1j*C4*eta*k**6*Log(2*Sqrt(k)) + 
             28350*C4*eta**2*k**7*Log(2*Sqrt(k)) + 3528000*C3*k**3*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             1499400*1j*C3*eta*k**4*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             520380*C4*k**5*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             117600*C3*eta**2*k**5*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             393225*1j*C4*eta*k**6*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             28350*C4*eta**2*k**7*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             3528000*C3*k**3*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             1499400*1j*C3*eta*k**4*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
             520380*C4*k**5*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             117600*C3*eta**2*k**5*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             393225*1j*C4*eta*k**6*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             28350*C4*eta**2*k**7*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             3528000*C3*k**3*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             1499400*1j*C3*eta*k**4*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) - 
             520380*C4*k**5*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             117600*C3*eta**2*k**5*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             393225*1j*C4*eta*k**6*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             28350*C4*eta**2*k**7*Log(2*(Sqrt(kmin) + Sqrt(k + kmin)))))/(2.8224e6*eta**2*k**2))
             
        return J_B
     

    def J_C(self, k, alpha, beta, C5, eta, kminoverride=None):
        """Solution for J_C"""
        kmax = k[-1]
        if kminoverride is not None:
            kmin = kminoverride
        else:
            kmin = k[0]
        
        J_C = (alpha ** 2 * C5 * (-(Sqrt(kmax * (-k + kmax)) * 
               (-28800 + 19200 * 1j * eta * (k - kmax) + 3840 * 1j * eta ** 3 * (k - kmax) ** 2 * kmax + 
                 80 * eta ** 2 * (69 * k ** 2 - 178 * k * kmax + 184 * kmax ** 2) + 
                 9 * eta ** 4 * (15 * k ** 4 + 10 * k ** 3 * kmax - 248 * k ** 2 * kmax ** 2 + 336 * k * kmax ** 3 - 128 * kmax ** 4) + 
                 400 * beta ** 2 * eta ** 2 * (-72 + 48 * 1j * eta * (k - kmax) + 
                    eta ** 2 * (3 * k ** 2 - 14 * k * kmax + 8 * kmax ** 2)) + 
                 320 * beta * eta * (-180 + 120 * 1j * eta * (k - kmax) + 12 * 1j * eta ** 3 * (k - kmax) ** 2 * kmax + 
                    eta ** 2 * (21 * k ** 2 - 62 * k * kmax + 56 * kmax ** 2)))) + 
            Sqrt(kmax * (k + kmax)) * (-28800 - 19200 * 1j * eta * (k + kmax) + 3840 * 1j * eta ** 3 * kmax * (k + kmax) ** 2 + 
               80 * eta ** 2 * (69 * k ** 2 + 178 * k * kmax + 184 * kmax ** 2) + 
               9 * eta ** 4 * (15 * k ** 4 - 10 * k ** 3 * kmax - 248 * k ** 2 * kmax ** 2 - 336 * k * kmax ** 3 - 128 * kmax ** 4) + 
               400 * beta ** 2 * eta ** 2 * (-72 - 48 * 1j * eta * (k + kmax) + eta ** 2 * (3 * k ** 2 + 14 * k * kmax + 8 * kmax ** 2)) + 
               320 * beta * eta * (-180 - 120 * 1j * eta * (k + kmax) + 12 * 1j * eta ** 3 * kmax * (k + kmax) ** 2 + 
                  eta ** 2 * (21 * k ** 2 + 62 * k * kmax + 56 * kmax ** 2))) + 
            Sqrt(kmin) * Sqrt(k + kmin) * (28800 + 19200 * 1j * eta * (k + kmin) - 
               3840 * 1j * eta ** 3 * kmin * (k + kmin) ** 2 - 80 * eta ** 2 * (69 * k ** 2 + 178 * k * kmin + 184 * kmin ** 2) + 
               9 * eta ** 4 * (-15 * k ** 4 + 10 * k ** 3 * kmin + 248 * k ** 2 * kmin ** 2 + 336 * k * kmin ** 3 + 128 * kmin ** 4) - 
               400 * beta ** 2 * eta ** 2 * (-72 - 48 * 1j * eta * (k + kmin) + eta ** 2 * (3 * k ** 2 + 14 * k * kmin + 8 * kmin ** 2)) + 
               320 * beta * eta * (180 + 120 * 1j * eta * (k + kmin) - 12 * 1j * eta ** 3 * kmin * (k + kmin) ** 2 - 
                  eta ** 2 * (21 * k ** 2 + 62 * k * kmin + 56 * kmin ** 2))) - 
            Sqrt((k - kmin) * kmin) * (28800 + 4800 * 1j * eta * (k + 2 * kmin) - 
               80 * eta ** 2 * (39 * k ** 2 - 38 * k * kmin + 104 * kmin ** 2) + 
               60 * 1j * eta ** 3 * (15 * k ** 3 - 54 * k ** 2 * kmin + 8 * k * kmin ** 2 + 16 * kmin ** 3) - 
               9 * eta ** 4 * (15 * k ** 4 + 10 * k ** 3 * kmin - 248 * k ** 2 * kmin ** 2 + 336 * k * kmin ** 3 - 128 * kmin ** 4) + 
               400 * beta ** 2 * eta ** 2 * (72 + 12 * 1j * eta * (k + 2 * kmin) + 
                  eta ** 2 * (3 * k ** 2 - 14 * k * kmin + 8 * kmin ** 2)) + 
               20 * 1j * beta * eta * (-2880 * 1j + 480 * eta * (k + 2 * kmin) + 
                  32 * 1j * eta ** 2 * (3 * k ** 2 + 4 * k * kmin + 8 * kmin ** 2) + 
                  3 * eta ** 3 * (15 * k ** 3 - 54 * k ** 2 * kmin + 8 * k * kmin ** 2 + 16 * kmin ** 3))) + 
            (15 * k * (1920 + 960 * 1j * eta * k - 560 * eta ** 2 * k ** 2 - 60 * 1j * eta ** 3 * k ** 3 + 9 * eta ** 4 * k ** 4 - 
                 80 * beta ** 2 * eta ** 2 * (-24 - 12 * 1j * eta * k + eta ** 2 * k ** 2) + 
                 20 * beta * eta * (192 + 96 * 1j * eta * k - 32 * eta ** 2 * k ** 2 - 3 * 1j * eta ** 3 * k ** 3)) * Pi) / 2. - 
            15 * k * (1920 + 960 * 1j * eta * k - 560 * eta ** 2 * k ** 2 - 60 * 1j * eta ** 3 * k ** 3 + 9 * eta ** 4 * k ** 4 - 
               80 * beta ** 2 * eta ** 2 * (-24 - 12 * 1j * eta * k + eta ** 2 * k ** 2) + 
               20 * beta * eta * (192 + 96 * 1j * eta * k - 32 * eta ** 2 * k ** 2 - 3 * 1j * eta ** 3 * k ** 3)) * 
             ArcTan(Sqrt(kmin / (k - kmin))) + 
            15 * k * (1920 - 400 * eta ** 2 * k ** 2 + 9 * eta ** 4 * k ** 4 - 320 * beta * eta * (-12 + eta ** 2 * k ** 2) + 
               80 * beta ** 2 * eta ** 2 * (24 + eta ** 2 * k ** 2)) * Log(2 * Sqrt(k)) - 
            15 * k * (1920 - 400 * eta ** 2 * k ** 2 + 9 * eta ** 4 * k ** 4 - 320 * beta * eta * (-12 + eta ** 2 * k ** 2) + 
               80 * beta ** 2 * eta ** 2 * (24 + eta ** 2 * k ** 2)) * Log(2 * (Sqrt(kmax) + Sqrt(-k + kmax))) - 
            15 * k * (1920 - 400 * eta ** 2 * k ** 2 + 9 * eta ** 4 * k ** 4 - 320 * beta * eta * (-12 + eta ** 2 * k ** 2) + 
               80 * beta ** 2 * eta ** 2 * (24 + eta ** 2 * k ** 2)) * Log(2 * (Sqrt(kmax) + Sqrt(k + kmax))) + 
            15 * k * (1920 - 400 * eta ** 2 * k ** 2 + 9 * eta ** 4 * k ** 4 - 320 * beta * eta * (-12 + eta ** 2 * k ** 2) + 
               80 * beta ** 2 * eta ** 2 * (24 + eta ** 2 * k ** 2)) * Log(2 * (Sqrt(kmin) + Sqrt(k + kmin))))) / (14400. * beta ** 2 * eta ** 4 * k)

        return J_C
    
    def J_D(self, k, alpha, beta, C6, C7, eta, kminoverride=None):
        """Solution for J_D"""
        kmax = k[-1]
        if kminoverride is not None:
            kmin = kminoverride
        else:
            kmin = k[0]
        
        j1 = (alpha**2*((483840*C6*k**2*Sqrt(-k + kmax))/kmax**1.5 - (967680*C6*k*Sqrt(-k + kmax))/Sqrt(kmax) - 
             725760*C6*Sqrt(kmax)*Sqrt(-k + kmax) - 695520*C7*k**2*Sqrt(kmax)*Sqrt(-k + kmax) + 
             181440*C7*k*kmax**1.5*Sqrt(-k + kmax) - 241920*C7*kmax**2.5*Sqrt(-k + kmax) - 
             (483840*C6*k**2*Sqrt(k + kmax))/kmax**1.5 - (967680*C6*k*Sqrt(k + kmax))/Sqrt(kmax) + 
             725760*C6*Sqrt(kmax)*Sqrt(k + kmax) + 695520*C7*k**2*Sqrt(kmax)*Sqrt(k + kmax) + 
             181440*C7*k*kmax**1.5*Sqrt(k + kmax) + 241920*C7*kmax**2.5*Sqrt(k + kmax) - 
             (483840*C6*k**2*Sqrt(k - kmin))/kmin**1.5 + (967680*C6*k*Sqrt(k - kmin))/Sqrt(kmin) + 
             725760*C6*Sqrt(k - kmin)*Sqrt(kmin) + 695520*C7*k**2*Sqrt(k - kmin)*Sqrt(kmin) - 
             181440*C7*k*Sqrt(k - kmin)*kmin**1.5 + 241920*C7*Sqrt(k - kmin)*kmin**2.5 + 
             (483840*C6*k**2*Sqrt(k + kmin))/kmin**1.5 + (967680*C6*k*Sqrt(k + kmin))/Sqrt(kmin) - 
             725760*C6*Sqrt(kmin)*Sqrt(k + kmin) - 695520*C7*k**2*Sqrt(kmin)*Sqrt(k + kmin) - 
             181440*C7*k*kmin**1.5*Sqrt(k + kmin) - 241920*C7*kmin**2.5*Sqrt(k + kmin) - 604800*C6*k*Pi - 
             378000*C7*k**3*Pi + 1209600*C6*k*ArcTan(Sqrt(kmin/(k - kmin))) + 
             756000*C7*k**3*ArcTan(Sqrt(kmin/(k - kmin))) - 1209600*C6*k*Log(2*Sqrt(k)) - 
             756000*C7*k**3*Log(2*Sqrt(k)) + 1209600*C6*k*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             756000*C7*k**3*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             1209600*C6*k*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
             756000*C7*k**3*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             1209600*C6*k*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) - 
             756000*C7*k**3*Log(2*(Sqrt(kmin) + Sqrt(k + kmin)))))/(604800.*beta**2*eta**4*k**2) 
        j2 = (alpha**2*((967680*beta*C6*k**2*Sqrt(-k + kmax))/kmax**1.5 + 
             (76800*1j*C6*k**3*Sqrt(-k + kmax))/kmax**1.5 - 
             (1935360*beta*C6*k*Sqrt(-k + kmax))/Sqrt(kmax) + 
             (1489920*1j*C6*k**2*Sqrt(-k + kmax))/Sqrt(kmax) - 
             1451520*beta*C6*Sqrt(kmax)*Sqrt(-k + kmax) + 452160*1j*C6*k*Sqrt(kmax)*Sqrt(-k + kmax) - 
             1391040*beta*C7*k**2*Sqrt(kmax)*Sqrt(-k + kmax) + 
             90900*1j*C7*k**3*Sqrt(kmax)*Sqrt(-k + kmax) - 305280*1j*C6*kmax**1.5*Sqrt(-k + kmax) + 
             362880*beta*C7*k*kmax**1.5*Sqrt(-k + kmax) - 384840*1j*C7*k**2*kmax**1.5*Sqrt(-k + kmax) - 
             483840*beta*C7*kmax**2.5*Sqrt(-k + kmax) + 125280*1j*C7*k*kmax**2.5*Sqrt(-k + kmax) - 
             152640*1j*C7*kmax**3.5*Sqrt(-k + kmax) - (967680*beta*C6*k**2*Sqrt(k + kmax))/kmax**1.5 + 
             (76800*1j*C6*k**3*Sqrt(k + kmax))/kmax**1.5 - (1935360*beta*C6*k*Sqrt(k + kmax))/Sqrt(kmax) - 
             (1489920*1j*C6*k**2*Sqrt(k + kmax))/Sqrt(kmax) + 1451520*beta*C6*Sqrt(kmax)*Sqrt(k + kmax) + 
             452160*1j*C6*k*Sqrt(kmax)*Sqrt(k + kmax) + 1391040*beta*C7*k**2*Sqrt(kmax)*Sqrt(k + kmax) + 
             90900*1j*C7*k**3*Sqrt(kmax)*Sqrt(k + kmax) + 305280*1j*C6*kmax**1.5*Sqrt(k + kmax) + 
             362880*beta*C7*k*kmax**1.5*Sqrt(k + kmax) + 384840*1j*C7*k**2*kmax**1.5*Sqrt(k + kmax) + 
             483840*beta*C7*kmax**2.5*Sqrt(k + kmax) + 125280*1j*C7*k*kmax**2.5*Sqrt(k + kmax) + 
             152640*1j*C7*kmax**3.5*Sqrt(k + kmax) - (967680*beta*C6*k**2*Sqrt(k - kmin))/kmin**1.5 + 
             (76800*1j*C6*k**3*Sqrt(k - kmin))/kmin**1.5 + (1935360*beta*C6*k*Sqrt(k - kmin))/Sqrt(kmin) - 
             (1413120*1j*C6*k**2*Sqrt(k - kmin))/Sqrt(kmin) + 1451520*beta*C6*Sqrt(k - kmin)*Sqrt(kmin) - 
             394560*1j*C6*k*Sqrt(k - kmin)*Sqrt(kmin) + 1391040*beta*C7*k**2*Sqrt(k - kmin)*Sqrt(kmin) - 
             324900*1j*C7*k**3*Sqrt(k - kmin)*Sqrt(kmin) + 420480*1j*C6*Sqrt(k - kmin)*kmin**1.5 - 
             362880*beta*C7*k*Sqrt(k - kmin)*kmin**1.5 + 305640*1j*C7*k**2*Sqrt(k - kmin)*kmin**1.5 + 
             483840*beta*C7*Sqrt(k - kmin)*kmin**2.5 - 96480*1j*C7*k*Sqrt(k - kmin)*kmin**2.5 + 
             210240*1j*C7*Sqrt(k - kmin)*kmin**3.5 + (967680*beta*C6*k**2*Sqrt(k + kmin))/kmin**1.5 - 
             (76800*1j*C6*k**3*Sqrt(k + kmin))/kmin**1.5 + (1935360*beta*C6*k*Sqrt(k + kmin))/Sqrt(kmin) + 
             (1489920*1j*C6*k**2*Sqrt(k + kmin))/Sqrt(kmin) - 1451520*beta*C6*Sqrt(kmin)*Sqrt(k + kmin) - 
             452160*1j*C6*k*Sqrt(kmin)*Sqrt(k + kmin) - 1391040*beta*C7*k**2*Sqrt(kmin)*Sqrt(k + kmin) - 
             90900*1j*C7*k**3*Sqrt(kmin)*Sqrt(k + kmin) - 305280*1j*C6*kmin**1.5*Sqrt(k + kmin) - 
             362880*beta*C7*k*kmin**1.5*Sqrt(k + kmin) - 384840*1j*C7*k**2*kmin**1.5*Sqrt(k + kmin) - 
             483840*beta*C7*kmin**2.5*Sqrt(k + kmin) - 125280*1j*C7*k*kmin**2.5*Sqrt(k + kmin) - 
             152640*1j*C7*kmin**3.5*Sqrt(k + kmin) - 1209600*beta*C6*k*Pi + 655200*1j*C6*k**2*Pi - 
             756000*beta*C7*k**3*Pi - 47250*1j*C7*k**4*Pi + 
             2419200*beta*C6*k*ArcTan(Sqrt(kmin/(k - kmin))) - 
             1310400*1j*C6*k**2*ArcTan(Sqrt(kmin/(k - kmin))) + 
             1512000*beta*C7*k**3*ArcTan(Sqrt(kmin/(k - kmin))) + 
             94500*1j*C7*k**4*ArcTan(Sqrt(kmin/(k - kmin))) - 2419200*beta*C6*k*Log(2*Sqrt(k)) + 
             1713600*1j*C6*k**2*Log(2*Sqrt(k)) - 1512000*beta*C7*k**3*Log(2*Sqrt(k)) - 
             321300*1j*C7*k**4*Log(2*Sqrt(k)) + 2419200*beta*C6*k*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             1713600*1j*C6*k**2*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             1512000*beta*C7*k**3*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             321300*1j*C7*k**4*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             2419200*beta*C6*k*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
             1713600*1j*C6*k**2*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
             1512000*beta*C7*k**3*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             321300*1j*C7*k**4*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             2419200*beta*C6*k*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) - 
             1713600*1j*C6*k**2*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) - 
             1512000*beta*C7*k**3*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             321300*1j*C7*k**4*Log(2*(Sqrt(kmin) + Sqrt(k + kmin)))))/(604800.*beta**2*eta**3*k**2) 
        j3 = (alpha**2*((483840*beta**2*C6*k**2*Sqrt(-k + kmax))/kmax**1.5 + 
             (153600*1j*beta*C6*k**3*Sqrt(-k + kmax))/kmax**1.5 + 
             (35840*C6*k**4*Sqrt(-k + kmax))/kmax**1.5 - (967680*beta**2*C6*k*Sqrt(-k + kmax))/Sqrt(kmax) + 
             (2979840*1j*beta*C6*k**2*Sqrt(-k + kmax))/Sqrt(kmax) - 
             (212480*C6*k**3*Sqrt(-k + kmax))/Sqrt(kmax) - 725760*beta**2*C6*Sqrt(kmax)*Sqrt(-k + kmax) + 
             904320*1j*beta*C6*k*Sqrt(kmax)*Sqrt(-k + kmax) + 956640*C6*k**2*Sqrt(kmax)*Sqrt(-k + kmax) - 
             695520*beta**2*C7*k**2*Sqrt(kmax)*Sqrt(-k + kmax) + 
             181800*1j*beta*C7*k**3*Sqrt(kmax)*Sqrt(-k + kmax) - 
             198870*C7*k**4*Sqrt(kmax)*Sqrt(-k + kmax) - 610560*1j*beta*C6*kmax**1.5*Sqrt(-k + kmax) - 
             209600*C6*k*kmax**1.5*Sqrt(-k + kmax) + 181440*beta**2*C7*k*kmax**1.5*Sqrt(-k + kmax) - 
             769680*1j*beta*C7*k**2*kmax**1.5*Sqrt(-k + kmax) - 37860*C7*k**3*kmax**1.5*Sqrt(-k + kmax) + 
             185600*C6*kmax**2.5*Sqrt(-k + kmax) - 241920*beta**2*C7*kmax**2.5*Sqrt(-k + kmax) + 
             250560*1j*beta*C7*k*kmax**2.5*Sqrt(-k + kmax) + 312240*C7*k**2*kmax**2.5*Sqrt(-k + kmax) - 
             305280*1j*beta*C7*kmax**3.5*Sqrt(-k + kmax) - 95520*C7*k*kmax**3.5*Sqrt(-k + kmax) + 
             111360*C7*kmax**4.5*Sqrt(-k + kmax) - (483840*beta**2*C6*k**2*Sqrt(k + kmax))/kmax**1.5 + 
             (153600*1j*beta*C6*k**3*Sqrt(k + kmax))/kmax**1.5 - 
             (35840*C6*k**4*Sqrt(k + kmax))/kmax**1.5 - (967680*beta**2*C6*k*Sqrt(k + kmax))/Sqrt(kmax) - 
             (2979840*1j*beta*C6*k**2*Sqrt(k + kmax))/Sqrt(kmax) - 
             (212480*C6*k**3*Sqrt(k + kmax))/Sqrt(kmax) + 725760*beta**2*C6*Sqrt(kmax)*Sqrt(k + kmax) + 
             904320*1j*beta*C6*k*Sqrt(kmax)*Sqrt(k + kmax) - 956640*C6*k**2*Sqrt(kmax)*Sqrt(k + kmax) + 
             695520*beta**2*C7*k**2*Sqrt(kmax)*Sqrt(k + kmax) + 
             181800*1j*beta*C7*k**3*Sqrt(kmax)*Sqrt(k + kmax) + 198870*C7*k**4*Sqrt(kmax)*Sqrt(k + kmax) + 
             610560*1j*beta*C6*kmax**1.5*Sqrt(k + kmax) - 209600*C6*k*kmax**1.5*Sqrt(k + kmax) + 
             181440*beta**2*C7*k*kmax**1.5*Sqrt(k + kmax) + 
             769680*1j*beta*C7*k**2*kmax**1.5*Sqrt(k + kmax) - 37860*C7*k**3*kmax**1.5*Sqrt(k + kmax) - 
             185600*C6*kmax**2.5*Sqrt(k + kmax) + 241920*beta**2*C7*kmax**2.5*Sqrt(k + kmax) + 
             250560*1j*beta*C7*k*kmax**2.5*Sqrt(k + kmax) - 312240*C7*k**2*kmax**2.5*Sqrt(k + kmax) + 
             305280*1j*beta*C7*kmax**3.5*Sqrt(k + kmax) - 95520*C7*k*kmax**3.5*Sqrt(k + kmax) - 
             111360*C7*kmax**4.5*Sqrt(k + kmax) - (483840*beta**2*C6*k**2*Sqrt(k - kmin))/kmin**1.5 + 
             (153600*1j*beta*C6*k**3*Sqrt(k - kmin))/kmin**1.5 - 
             (35840*C6*k**4*Sqrt(k - kmin))/kmin**1.5 + (967680*beta**2*C6*k*Sqrt(k - kmin))/Sqrt(kmin) - 
             (2826240*1j*beta*C6*k**2*Sqrt(k - kmin))/Sqrt(kmin) - 
             (248320*C6*k**3*Sqrt(k - kmin))/Sqrt(kmin) + 725760*beta**2*C6*Sqrt(k - kmin)*Sqrt(kmin) - 
             789120*1j*beta*C6*k*Sqrt(k - kmin)*Sqrt(kmin) - 783840*C6*k**2*Sqrt(k - kmin)*Sqrt(kmin) + 
             695520*beta**2*C7*k**2*Sqrt(k - kmin)*Sqrt(kmin) - 
             649800*1j*beta*C7*k**3*Sqrt(k - kmin)*Sqrt(kmin) + 148470*C7*k**4*Sqrt(k - kmin)*Sqrt(kmin) + 
             840960*1j*beta*C6*Sqrt(k - kmin)*kmin**1.5 + 171200*C6*k*Sqrt(k - kmin)*kmin**1.5 - 
             181440*beta**2*C7*k*Sqrt(k - kmin)*kmin**1.5 + 
             611280*1j*beta*C7*k**2*Sqrt(k - kmin)*kmin**1.5 + 157860*C7*k**3*Sqrt(k - kmin)*kmin**1.5 - 
             262400*C6*Sqrt(k - kmin)*kmin**2.5 + 241920*beta**2*C7*Sqrt(k - kmin)*kmin**2.5 - 
             192960*1j*beta*C7*k*Sqrt(k - kmin)*kmin**2.5 - 262320*C7*k**2*Sqrt(k - kmin)*kmin**2.5 + 
             420480*1j*beta*C7*Sqrt(k - kmin)*kmin**3.5 + 72480*C7*k*Sqrt(k - kmin)*kmin**3.5 - 
             157440*C7*Sqrt(k - kmin)*kmin**4.5 + (483840*beta**2*C6*k**2*Sqrt(k + kmin))/kmin**1.5 - 
             (153600*1j*beta*C6*k**3*Sqrt(k + kmin))/kmin**1.5 + 
             (35840*C6*k**4*Sqrt(k + kmin))/kmin**1.5 + (967680*beta**2*C6*k*Sqrt(k + kmin))/Sqrt(kmin) + 
             (2979840*1j*beta*C6*k**2*Sqrt(k + kmin))/Sqrt(kmin) + 
             (212480*C6*k**3*Sqrt(k + kmin))/Sqrt(kmin) - 725760*beta**2*C6*Sqrt(kmin)*Sqrt(k + kmin) - 
             904320*1j*beta*C6*k*Sqrt(kmin)*Sqrt(k + kmin) + 956640*C6*k**2*Sqrt(kmin)*Sqrt(k + kmin) - 
             695520*beta**2*C7*k**2*Sqrt(kmin)*Sqrt(k + kmin) - 
             181800*1j*beta*C7*k**3*Sqrt(kmin)*Sqrt(k + kmin) - 198870*C7*k**4*Sqrt(kmin)*Sqrt(k + kmin) - 
             610560*1j*beta*C6*kmin**1.5*Sqrt(k + kmin) + 209600*C6*k*kmin**1.5*Sqrt(k + kmin) - 
             181440*beta**2*C7*k*kmin**1.5*Sqrt(k + kmin) - 
             769680*1j*beta*C7*k**2*kmin**1.5*Sqrt(k + kmin) + 37860*C7*k**3*kmin**1.5*Sqrt(k + kmin) + 
             185600*C6*kmin**2.5*Sqrt(k + kmin) - 241920*beta**2*C7*kmin**2.5*Sqrt(k + kmin) - 
             250560*1j*beta*C7*k*kmin**2.5*Sqrt(k + kmin) + 312240*C7*k**2*kmin**2.5*Sqrt(k + kmin) - 
             305280*1j*beta*C7*kmin**3.5*Sqrt(k + kmin) + 95520*C7*k*kmin**3.5*Sqrt(k + kmin) + 
             111360*C7*kmin**4.5*Sqrt(k + kmin) - 604800*beta**2*C6*k*Pi + 1310400*1j*beta*C6*k**2*Pi + 
             579600*C6*k**3*Pi - 378000*beta**2*C7*k**3*Pi - 94500*1j*beta*C7*k**4*Pi + 20475*C7*k**5*Pi + 
             1209600*beta**2*C6*k*ArcTan(Sqrt(kmin/(k - kmin))) - 
             2620800*1j*beta*C6*k**2*ArcTan(Sqrt(kmin/(k - kmin))) - 
             1159200*C6*k**3*ArcTan(Sqrt(kmin/(k - kmin))) + 
             756000*beta**2*C7*k**3*ArcTan(Sqrt(kmin/(k - kmin))) + 
             189000*1j*beta*C7*k**4*ArcTan(Sqrt(kmin/(k - kmin))) - 
             40950*C7*k**5*ArcTan(Sqrt(kmin/(k - kmin))) - 1209600*beta**2*C6*k*Log(2*Sqrt(k)) + 
             3427200*1j*beta*C6*k**2*Log(2*Sqrt(k)) + 756000*C6*k**3*Log(2*Sqrt(k)) - 
             756000*beta**2*C7*k**3*Log(2*Sqrt(k)) - 642600*1j*beta*C7*k**4*Log(2*Sqrt(k)) + 
             91350*C7*k**5*Log(2*Sqrt(k)) + 1209600*beta**2*C6*k*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             3427200*1j*beta*C6*k**2*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             756000*C6*k**3*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             756000*beta**2*C7*k**3*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             642600*1j*beta*C7*k**4*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             91350*C7*k**5*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             1209600*beta**2*C6*k*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
             3427200*1j*beta*C6*k**2*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             756000*C6*k**3*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
             756000*beta**2*C7*k**3*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             642600*1j*beta*C7*k**4*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             91350*C7*k**5*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             1209600*beta**2*C6*k*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) - 
             3427200*1j*beta*C6*k**2*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             756000*C6*k**3*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) - 
             756000*beta**2*C7*k**3*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             642600*1j*beta*C7*k**4*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             91350*C7*k**5*Log(2*(Sqrt(kmin) + Sqrt(k + kmin)))))/(604800.*beta**2*eta**2*k**2) 
        j4= (alpha**2*((76800*1j*beta**2*C6*k**3*Sqrt(-k + kmax))/kmax**1.5 + 
             (35840*beta*C6*k**4*Sqrt(-k + kmax))/kmax**1.5 + 
             (1489920*1j*beta**2*C6*k**2*Sqrt(-k + kmax))/Sqrt(kmax) - 
             (442880*beta*C6*k**3*Sqrt(-k + kmax))/Sqrt(kmax) + 
             (107520*1j*C6*k**4*Sqrt(-k + kmax))/Sqrt(kmax) + 
             452160*1j*beta**2*C6*k*Sqrt(kmax)*Sqrt(-k + kmax) + 
             1043040*beta*C6*k**2*Sqrt(kmax)*Sqrt(-k + kmax) + 
             57360*1j*C6*k**3*Sqrt(kmax)*Sqrt(-k + kmax) + 
             90900*1j*beta**2*C7*k**3*Sqrt(kmax)*Sqrt(-k + kmax) - 
             224070*beta*C7*k**4*Sqrt(kmax)*Sqrt(-k + kmax) - 3150*1j*C7*k**5*Sqrt(kmax)*Sqrt(-k + kmax) - 
             305280*1j*beta**2*C6*kmax**1.5*Sqrt(-k + kmax) - 228800*beta*C6*k*kmax**1.5*Sqrt(-k + kmax) + 
             125280*1j*C6*k**2*kmax**1.5*Sqrt(-k + kmax) - 
             384840*1j*beta**2*C7*k**2*kmax**1.5*Sqrt(-k + kmax) + 
             22140*beta*C7*k**3*kmax**1.5*Sqrt(-k + kmax) - 37940*1j*C7*k**4*kmax**1.5*Sqrt(-k + kmax) + 
             147200*beta*C6*kmax**2.5*Sqrt(-k + kmax) - 21120*1j*C6*k*kmax**2.5*Sqrt(-k + kmax) + 
             125280*1j*beta**2*C7*k*kmax**2.5*Sqrt(-k + kmax) + 
             337200*beta*C7*k**2*kmax**2.5*Sqrt(-k + kmax) + 26480*1j*C7*k**3*kmax**2.5*Sqrt(-k + kmax) - 
             42240*1j*C6*kmax**3.5*Sqrt(-k + kmax) - 152640*1j*beta**2*C7*kmax**3.5*Sqrt(-k + kmax) - 
             107040*beta*C7*k*kmax**3.5*Sqrt(-k + kmax) + 60000*1j*C7*k**2*kmax**3.5*Sqrt(-k + kmax) + 
             88320*beta*C7*kmax**4.5*Sqrt(-k + kmax) - 14080*1j*C7*k*kmax**4.5*Sqrt(-k + kmax) - 
             28160*1j*C7*kmax**5.5*Sqrt(-k + kmax) + 
             (76800*1j*beta**2*C6*k**3*Sqrt(k + kmax))/kmax**1.5 - 
             (35840*beta*C6*k**4*Sqrt(k + kmax))/kmax**1.5 - 
             (1489920*1j*beta**2*C6*k**2*Sqrt(k + kmax))/Sqrt(kmax) - 
             (442880*beta*C6*k**3*Sqrt(k + kmax))/Sqrt(kmax) - 
             (107520*1j*C6*k**4*Sqrt(k + kmax))/Sqrt(kmax) + 
             452160*1j*beta**2*C6*k*Sqrt(kmax)*Sqrt(k + kmax) - 
             1043040*beta*C6*k**2*Sqrt(kmax)*Sqrt(k + kmax) + 57360*1j*C6*k**3*Sqrt(kmax)*Sqrt(k + kmax) + 
             90900*1j*beta**2*C7*k**3*Sqrt(kmax)*Sqrt(k + kmax) + 
             224070*beta*C7*k**4*Sqrt(kmax)*Sqrt(k + kmax) - 3150*1j*C7*k**5*Sqrt(kmax)*Sqrt(k + kmax) + 
             305280*1j*beta**2*C6*kmax**1.5*Sqrt(k + kmax) - 228800*beta*C6*k*kmax**1.5*Sqrt(k + kmax) - 
             125280*1j*C6*k**2*kmax**1.5*Sqrt(k + kmax) + 
             384840*1j*beta**2*C7*k**2*kmax**1.5*Sqrt(k + kmax) + 
             22140*beta*C7*k**3*kmax**1.5*Sqrt(k + kmax) + 37940*1j*C7*k**4*kmax**1.5*Sqrt(k + kmax) - 
             147200*beta*C6*kmax**2.5*Sqrt(k + kmax) - 21120*1j*C6*k*kmax**2.5*Sqrt(k + kmax) + 
             125280*1j*beta**2*C7*k*kmax**2.5*Sqrt(k + kmax) - 
             337200*beta*C7*k**2*kmax**2.5*Sqrt(k + kmax) + 26480*1j*C7*k**3*kmax**2.5*Sqrt(k + kmax) + 
             42240*1j*C6*kmax**3.5*Sqrt(k + kmax) + 152640*1j*beta**2*C7*kmax**3.5*Sqrt(k + kmax) - 
             107040*beta*C7*k*kmax**3.5*Sqrt(k + kmax) - 60000*1j*C7*k**2*kmax**3.5*Sqrt(k + kmax) - 
             88320*beta*C7*kmax**4.5*Sqrt(k + kmax) - 14080*1j*C7*k*kmax**4.5*Sqrt(k + kmax) + 
             28160*1j*C7*kmax**5.5*Sqrt(k + kmax) + (76800*1j*beta**2*C6*k**3*Sqrt(k - kmin))/kmin**1.5 - 
             (35840*beta*C6*k**4*Sqrt(k - kmin))/kmin**1.5 - 
             (1413120*1j*beta**2*C6*k**2*Sqrt(k - kmin))/Sqrt(kmin) - 
             (478720*beta*C6*k**3*Sqrt(k - kmin))/Sqrt(kmin) - 
             (107520*1j*C6*k**4*Sqrt(k - kmin))/Sqrt(kmin) - 
             394560*1j*beta**2*C6*k*Sqrt(k - kmin)*Sqrt(kmin) - 
             697440*beta*C6*k**2*Sqrt(k - kmin)*Sqrt(kmin) + 176640*1j*C6*k**3*Sqrt(k - kmin)*Sqrt(kmin) - 
             324900*1j*beta**2*C7*k**3*Sqrt(k - kmin)*Sqrt(kmin) + 
             123270*beta*C7*k**4*Sqrt(k - kmin)*Sqrt(kmin) - 18900*1j*C7*k**5*Sqrt(k - kmin)*Sqrt(kmin) + 
             420480*1j*beta**2*C6*Sqrt(k - kmin)*kmin**1.5 + 152000*beta*C6*k*Sqrt(k - kmin)*kmin**1.5 - 
             46080*1j*C6*k**2*Sqrt(k - kmin)*kmin**1.5 + 
             305640*1j*beta**2*C7*k**2*Sqrt(k - kmin)*kmin**1.5 + 
             217860*beta*C7*k**3*Sqrt(k - kmin)*kmin**1.5 + 23240*1j*C7*k**4*Sqrt(k - kmin)*kmin**1.5 - 
             300800*beta*C6*Sqrt(k - kmin)*kmin**2.5 - 7680*1j*C6*k*Sqrt(k - kmin)*kmin**2.5 - 
             96480*1j*beta**2*C7*k*Sqrt(k - kmin)*kmin**2.5 - 
             237360*beta*C7*k**2*Sqrt(k - kmin)*kmin**2.5 + 53920*1j*C7*k**3*Sqrt(k - kmin)*kmin**2.5 - 
             15360*1j*C6*Sqrt(k - kmin)*kmin**3.5 + 210240*1j*beta**2*C7*Sqrt(k - kmin)*kmin**3.5 + 
             60960*beta*C7*k*Sqrt(k - kmin)*kmin**3.5 - 24000*1j*C7*k**2*Sqrt(k - kmin)*kmin**3.5 - 
             180480*beta*C7*Sqrt(k - kmin)*kmin**4.5 - 5120*1j*C7*k*Sqrt(k - kmin)*kmin**4.5 - 
             10240*1j*C7*Sqrt(k - kmin)*kmin**5.5 - (76800*1j*beta**2*C6*k**3*Sqrt(k + kmin))/kmin**1.5 + 
             (35840*beta*C6*k**4*Sqrt(k + kmin))/kmin**1.5 + 
             (1489920*1j*beta**2*C6*k**2*Sqrt(k + kmin))/Sqrt(kmin) + 
             (442880*beta*C6*k**3*Sqrt(k + kmin))/Sqrt(kmin) + 
             (107520*1j*C6*k**4*Sqrt(k + kmin))/Sqrt(kmin) - 
             452160*1j*beta**2*C6*k*Sqrt(kmin)*Sqrt(k + kmin) + 
             1043040*beta*C6*k**2*Sqrt(kmin)*Sqrt(k + kmin) - 57360*1j*C6*k**3*Sqrt(kmin)*Sqrt(k + kmin) - 
             90900*1j*beta**2*C7*k**3*Sqrt(kmin)*Sqrt(k + kmin) - 
             224070*beta*C7*k**4*Sqrt(kmin)*Sqrt(k + kmin) + 3150*1j*C7*k**5*Sqrt(kmin)*Sqrt(k + kmin) - 
             305280*1j*beta**2*C6*kmin**1.5*Sqrt(k + kmin) + 228800*beta*C6*k*kmin**1.5*Sqrt(k + kmin) + 
             125280*1j*C6*k**2*kmin**1.5*Sqrt(k + kmin) - 
             384840*1j*beta**2*C7*k**2*kmin**1.5*Sqrt(k + kmin) - 
             22140*beta*C7*k**3*kmin**1.5*Sqrt(k + kmin) - 37940*1j*C7*k**4*kmin**1.5*Sqrt(k + kmin) + 
             147200*beta*C6*kmin**2.5*Sqrt(k + kmin) + 21120*1j*C6*k*kmin**2.5*Sqrt(k + kmin) - 
             125280*1j*beta**2*C7*k*kmin**2.5*Sqrt(k + kmin) + 
             337200*beta*C7*k**2*kmin**2.5*Sqrt(k + kmin) - 26480*1j*C7*k**3*kmin**2.5*Sqrt(k + kmin) - 
             42240*1j*C6*kmin**3.5*Sqrt(k + kmin) - 152640*1j*beta**2*C7*kmin**3.5*Sqrt(k + kmin) + 
             107040*beta*C7*k*kmin**3.5*Sqrt(k + kmin) + 60000*1j*C7*k**2*kmin**3.5*Sqrt(k + kmin) + 
             88320*beta*C7*kmin**4.5*Sqrt(k + kmin) + 14080*1j*C7*k*kmin**4.5*Sqrt(k + kmin) - 
             28160*1j*C7*kmin**5.5*Sqrt(k + kmin) + 655200*1j*beta**2*C6*k**2*Pi + 
             680400*beta*C6*k**3*Pi - 47250*1j*beta**2*C7*k**4*Pi + 7875*beta*C7*k**5*Pi - 
             9450*1j*C7*k**6*Pi - 1310400*1j*beta**2*C6*k**2*ArcTan(Sqrt(kmin/(k - kmin))) - 
             1360800*beta*C6*k**3*ArcTan(Sqrt(kmin/(k - kmin))) + 
             94500*1j*beta**2*C7*k**4*ArcTan(Sqrt(kmin/(k - kmin))) - 
             15750*beta*C7*k**5*ArcTan(Sqrt(kmin/(k - kmin))) + 
             18900*1j*C7*k**6*ArcTan(Sqrt(kmin/(k - kmin))) + 1713600*1j*beta**2*C6*k**2*Log(2*Sqrt(k)) + 
             554400*beta*C6*k**3*Log(2*Sqrt(k)) + 226800*1j*C6*k**4*Log(2*Sqrt(k)) - 
             321300*1j*beta**2*C7*k**4*Log(2*Sqrt(k)) + 116550*beta*C7*k**5*Log(2*Sqrt(k)) + 
             3150*1j*C7*k**6*Log(2*Sqrt(k)) - 
             1713600*1j*beta**2*C6*k**2*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             554400*beta*C6*k**3*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             226800*1j*C6*k**4*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             321300*1j*beta**2*C7*k**4*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             116550*beta*C7*k**5*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             3150*1j*C7*k**6*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             1713600*1j*beta**2*C6*k**2*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             554400*beta*C6*k**3*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
             226800*1j*C6*k**4*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             321300*1j*beta**2*C7*k**4*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             116550*beta*C7*k**5*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
             3150*1j*C7*k**6*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             1713600*1j*beta**2*C6*k**2*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             554400*beta*C6*k**3*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) - 
             226800*1j*C6*k**4*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             321300*1j*beta**2*C7*k**4*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             116550*beta*C7*k**5*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) - 
             3150*1j*C7*k**6*Log(2*(Sqrt(kmin) + Sqrt(k + kmin)))))/(604800.*beta**2*eta*k**2) 
        j5 = (alpha**2*((-230400*beta**2*C6*k**3*Sqrt(-k + kmax))/Sqrt(kmax) + 
             (107520*1j*beta*C6*k**4*Sqrt(-k + kmax))/Sqrt(kmax) + 
             86400*beta**2*C6*k**2*Sqrt(kmax)*Sqrt(-k + kmax) + 
             57360*1j*beta*C6*k**3*Sqrt(kmax)*Sqrt(-k + kmax) + 62160*C6*k**4*Sqrt(kmax)*Sqrt(-k + kmax) - 
             25200*beta**2*C7*k**4*Sqrt(kmax)*Sqrt(-k + kmax) - 
             3150*1j*beta*C7*k**5*Sqrt(kmax)*Sqrt(-k + kmax) - 2835*C7*k**6*Sqrt(kmax)*Sqrt(-k + kmax) - 
             19200*beta**2*C6*k*kmax**1.5*Sqrt(-k + kmax) + 
             125280*1j*beta*C6*k**2*kmax**1.5*Sqrt(-k + kmax) + 23520*C6*k**3*kmax**1.5*Sqrt(-k + kmax) + 
             60000*beta**2*C7*k**3*kmax**1.5*Sqrt(-k + kmax) - 
             37940*1j*beta*C7*k**4*kmax**1.5*Sqrt(-k + kmax) - 1890*C7*k**5*kmax**1.5*Sqrt(-k + kmax) - 
             38400*beta**2*C6*kmax**2.5*Sqrt(-k + kmax) - 21120*1j*beta*C6*k*kmax**2.5*Sqrt(-k + kmax) - 
             56448*C6*k**2*kmax**2.5*Sqrt(-k + kmax) + 24960*beta**2*C7*k**2*kmax**2.5*Sqrt(-k + kmax) + 
             26480*1j*beta*C7*k**3*kmax**2.5*Sqrt(-k + kmax) + 19992*C7*k**4*kmax**2.5*Sqrt(-k + kmax) - 
             42240*1j*beta*C6*kmax**3.5*Sqrt(-k + kmax) + 5376*C6*k*kmax**3.5*Sqrt(-k + kmax) - 
             11520*beta**2*C7*k*kmax**3.5*Sqrt(-k + kmax) + 
             60000*1j*beta*C7*k**2*kmax**3.5*Sqrt(-k + kmax) + 9456*C7*k**3*kmax**3.5*Sqrt(-k + kmax) + 
             10752*C6*kmax**4.5*Sqrt(-k + kmax) - 23040*beta**2*C7*kmax**4.5*Sqrt(-k + kmax) - 
             14080*1j*beta*C7*k*kmax**4.5*Sqrt(-k + kmax) - 33408*C7*k**2*kmax**4.5*Sqrt(-k + kmax) - 
             28160*1j*beta*C7*kmax**5.5*Sqrt(-k + kmax) + 3840*C7*k*kmax**5.5*Sqrt(-k + kmax) + 
             7680*C7*kmax**6.5*Sqrt(-k + kmax) - (230400*beta**2*C6*k**3*Sqrt(k + kmax))/Sqrt(kmax) - 
             (107520*1j*beta*C6*k**4*Sqrt(k + kmax))/Sqrt(kmax) - 
             86400*beta**2*C6*k**2*Sqrt(kmax)*Sqrt(k + kmax) + 
             57360*1j*beta*C6*k**3*Sqrt(kmax)*Sqrt(k + kmax) - 62160*C6*k**4*Sqrt(kmax)*Sqrt(k + kmax) + 
             25200*beta**2*C7*k**4*Sqrt(kmax)*Sqrt(k + kmax) - 
             3150*1j*beta*C7*k**5*Sqrt(kmax)*Sqrt(k + kmax) + 2835*C7*k**6*Sqrt(kmax)*Sqrt(k + kmax) - 
             19200*beta**2*C6*k*kmax**1.5*Sqrt(k + kmax) - 
             125280*1j*beta*C6*k**2*kmax**1.5*Sqrt(k + kmax) + 23520*C6*k**3*kmax**1.5*Sqrt(k + kmax) + 
             60000*beta**2*C7*k**3*kmax**1.5*Sqrt(k + kmax) + 
             37940*1j*beta*C7*k**4*kmax**1.5*Sqrt(k + kmax) - 1890*C7*k**5*kmax**1.5*Sqrt(k + kmax) + 
             38400*beta**2*C6*kmax**2.5*Sqrt(k + kmax) - 21120*1j*beta*C6*k*kmax**2.5*Sqrt(k + kmax) + 
             56448*C6*k**2*kmax**2.5*Sqrt(k + kmax) - 24960*beta**2*C7*k**2*kmax**2.5*Sqrt(k + kmax) + 
             26480*1j*beta*C7*k**3*kmax**2.5*Sqrt(k + kmax) - 19992*C7*k**4*kmax**2.5*Sqrt(k + kmax) + 
             42240*1j*beta*C6*kmax**3.5*Sqrt(k + kmax) + 5376*C6*k*kmax**3.5*Sqrt(k + kmax) - 
             11520*beta**2*C7*k*kmax**3.5*Sqrt(k + kmax) - 
             60000*1j*beta*C7*k**2*kmax**3.5*Sqrt(k + kmax) + 9456*C7*k**3*kmax**3.5*Sqrt(k + kmax) - 
             10752*C6*kmax**4.5*Sqrt(k + kmax) + 23040*beta**2*C7*kmax**4.5*Sqrt(k + kmax) - 
             14080*1j*beta*C7*k*kmax**4.5*Sqrt(k + kmax) + 33408*C7*k**2*kmax**4.5*Sqrt(k + kmax) + 
             28160*1j*beta*C7*kmax**5.5*Sqrt(k + kmax) + 3840*C7*k*kmax**5.5*Sqrt(k + kmax) - 
             7680*C7*kmax**6.5*Sqrt(k + kmax) - (230400*beta**2*C6*k**3*Sqrt(k - kmin))/Sqrt(kmin) - 
             (107520*1j*beta*C6*k**4*Sqrt(k - kmin))/Sqrt(kmin) + 
             86400*beta**2*C6*k**2*Sqrt(k - kmin)*Sqrt(kmin) + 
             176640*1j*beta*C6*k**3*Sqrt(k - kmin)*Sqrt(kmin) - 62160*C6*k**4*Sqrt(k - kmin)*Sqrt(kmin) - 
             25200*beta**2*C7*k**4*Sqrt(k - kmin)*Sqrt(kmin) - 
             18900*1j*beta*C7*k**5*Sqrt(k - kmin)*Sqrt(kmin) + 2835*C7*k**6*Sqrt(k - kmin)*Sqrt(kmin) - 
             19200*beta**2*C6*k*Sqrt(k - kmin)*kmin**1.5 - 
             46080*1j*beta*C6*k**2*Sqrt(k - kmin)*kmin**1.5 - 23520*C6*k**3*Sqrt(k - kmin)*kmin**1.5 + 
             60000*beta**2*C7*k**3*Sqrt(k - kmin)*kmin**1.5 + 
             23240*1j*beta*C7*k**4*Sqrt(k - kmin)*kmin**1.5 + 1890*C7*k**5*Sqrt(k - kmin)*kmin**1.5 - 
             38400*beta**2*C6*Sqrt(k - kmin)*kmin**2.5 - 7680*1j*beta*C6*k*Sqrt(k - kmin)*kmin**2.5 + 
             56448*C6*k**2*Sqrt(k - kmin)*kmin**2.5 + 24960*beta**2*C7*k**2*Sqrt(k - kmin)*kmin**2.5 + 
             53920*1j*beta*C7*k**3*Sqrt(k - kmin)*kmin**2.5 - 19992*C7*k**4*Sqrt(k - kmin)*kmin**2.5 - 
             15360*1j*beta*C6*Sqrt(k - kmin)*kmin**3.5 - 5376*C6*k*Sqrt(k - kmin)*kmin**3.5 - 
             11520*beta**2*C7*k*Sqrt(k - kmin)*kmin**3.5 - 
             24000*1j*beta*C7*k**2*Sqrt(k - kmin)*kmin**3.5 - 9456*C7*k**3*Sqrt(k - kmin)*kmin**3.5 - 
             10752*C6*Sqrt(k - kmin)*kmin**4.5 - 23040*beta**2*C7*Sqrt(k - kmin)*kmin**4.5 - 
             5120*1j*beta*C7*k*Sqrt(k - kmin)*kmin**4.5 + 33408*C7*k**2*Sqrt(k - kmin)*kmin**4.5 - 
             10240*1j*beta*C7*Sqrt(k - kmin)*kmin**5.5 - 3840*C7*k*Sqrt(k - kmin)*kmin**5.5 - 
             7680*C7*Sqrt(k - kmin)*kmin**6.5 + (230400*beta**2*C6*k**3*Sqrt(k + kmin))/Sqrt(kmin) + 
             (107520*1j*beta*C6*k**4*Sqrt(k + kmin))/Sqrt(kmin) + 
             86400*beta**2*C6*k**2*Sqrt(kmin)*Sqrt(k + kmin) - 
             57360*1j*beta*C6*k**3*Sqrt(kmin)*Sqrt(k + kmin) + 62160*C6*k**4*Sqrt(kmin)*Sqrt(k + kmin) - 
             25200*beta**2*C7*k**4*Sqrt(kmin)*Sqrt(k + kmin) + 
             3150*1j*beta*C7*k**5*Sqrt(kmin)*Sqrt(k + kmin) - 2835*C7*k**6*Sqrt(kmin)*Sqrt(k + kmin) + 
             19200*beta**2*C6*k*kmin**1.5*Sqrt(k + kmin) + 
             125280*1j*beta*C6*k**2*kmin**1.5*Sqrt(k + kmin) - 23520*C6*k**3*kmin**1.5*Sqrt(k + kmin) - 
             60000*beta**2*C7*k**3*kmin**1.5*Sqrt(k + kmin) - 
             37940*1j*beta*C7*k**4*kmin**1.5*Sqrt(k + kmin) + 1890*C7*k**5*kmin**1.5*Sqrt(k + kmin) - 
             38400*beta**2*C6*kmin**2.5*Sqrt(k + kmin) + 21120*1j*beta*C6*k*kmin**2.5*Sqrt(k + kmin) - 
             56448*C6*k**2*kmin**2.5*Sqrt(k + kmin) + 24960*beta**2*C7*k**2*kmin**2.5*Sqrt(k + kmin) - 
             26480*1j*beta*C7*k**3*kmin**2.5*Sqrt(k + kmin) + 19992*C7*k**4*kmin**2.5*Sqrt(k + kmin) - 
             42240*1j*beta*C6*kmin**3.5*Sqrt(k + kmin) - 5376*C6*k*kmin**3.5*Sqrt(k + kmin) + 
             11520*beta**2*C7*k*kmin**3.5*Sqrt(k + kmin) + 
             60000*1j*beta*C7*k**2*kmin**3.5*Sqrt(k + kmin) - 9456*C7*k**3*kmin**3.5*Sqrt(k + kmin) + 
             10752*C6*kmin**4.5*Sqrt(k + kmin) - 23040*beta**2*C7*kmin**4.5*Sqrt(k + kmin) + 
             14080*1j*beta*C7*k*kmin**4.5*Sqrt(k + kmin) - 33408*C7*k**2*kmin**4.5*Sqrt(k + kmin) - 
             28160*1j*beta*C7*kmin**5.5*Sqrt(k + kmin) - 3840*C7*k*kmin**5.5*Sqrt(k + kmin) + 
             7680*C7*kmin**6.5*Sqrt(k + kmin) + 100800*beta**2*C6*k**3*Pi + 22680*C6*k**5*Pi - 
             12600*beta**2*C7*k**5*Pi - 9450*1j*beta*C7*k**6*Pi + (2835*C7*k**7*Pi)/2. - 
             201600*beta**2*C6*k**3*ArcTan(Sqrt(kmin/(k - kmin))) - 
             45360*C6*k**5*ArcTan(Sqrt(kmin/(k - kmin))) + 
             25200*beta**2*C7*k**5*ArcTan(Sqrt(kmin/(k - kmin))) + 
             18900*1j*beta*C7*k**6*ArcTan(Sqrt(kmin/(k - kmin))) - 
             2835*C7*k**7*ArcTan(Sqrt(kmin/(k - kmin))) - 201600*beta**2*C6*k**3*Log(2*Sqrt(k)) + 
             226800*1j*beta*C6*k**4*Log(2*Sqrt(k)) + 45360*C6*k**5*Log(2*Sqrt(k)) + 
             25200*beta**2*C7*k**5*Log(2*Sqrt(k)) + 3150*1j*beta*C7*k**6*Log(2*Sqrt(k)) + 
             2835*C7*k**7*Log(2*Sqrt(k)) + 201600*beta**2*C6*k**3*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             226800*1j*beta*C6*k**4*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             45360*C6*k**5*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             25200*beta**2*C7*k**5*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             3150*1j*beta*C7*k**6*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) - 
             2835*C7*k**7*Log(2*(Sqrt(kmax) + Sqrt(-k + kmax))) + 
             201600*beta**2*C6*k**3*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
             226800*1j*beta*C6*k**4*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             45360*C6*k**5*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             25200*beta**2*C7*k**5*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) + 
             3150*1j*beta*C7*k**6*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             2835*C7*k**7*Log(2*(Sqrt(kmax) + Sqrt(k + kmax))) - 
             201600*beta**2*C6*k**3*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) - 
             226800*1j*beta*C6*k**4*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             45360*C6*k**5*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             25200*beta**2*C7*k**5*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) - 
             3150*1j*beta*C7*k**6*Log(2*(Sqrt(kmin) + Sqrt(k + kmin))) + 
             2835*C7*k**7*Log(2*(Sqrt(kmin) + Sqrt(k + kmin)))))/(604800.*beta**2*k**2)
             
        return j1 + j2 + j3 + j4 + j5
    def get_vars_from_model(self, m, nix):
        """Find and calculate variables from cosmomodels model."""
        try:
            #Get background values
            phi, phidot, H = m.yresult[nix, 0:3, 0]
            a = m.ainit*np.exp(m.tresult[nix])
        except AttributeError:
            raise
        
        if np.any(np.isnan(phi)):
            raise AttributeError("Background values not available for this timestep.")
        
        #Get potentials
        V, Vp, Vpp, Vppp = m.potentials(np.array([phi]))
        
        #Set alpha and beta
        alpha = 1/(a*np.sqrt(2))
        beta = a*H
        
        
        eta = -1/(beta*(1-m.bgepsilon[nix]))
        
        k = self.srceqns.k
        #Set ones array with same shape as self.k
        onekshape = np.ones(k.shape)
        
        #Set C_i values
        C1 = 1/H**2 * (Vppp + phidot/a**2 * (3 * a**2 * Vpp + 2 * k**2 ))
        
        C2 = 3.5 * phidot /((a*H)**2) * onekshape
        
        C3 = -4.5 / (a*H**2) * k
        
        C4 = -phidot/(a*H**2) / k
        
        C5 = -1.5 * phidot * onekshape
        
        C6 = 2 * phidot * k
        
        C7 = - phidot / k
        
        return (alpha, beta), (C1, C2, C3, C4, C5, C6, C7), eta
    
    def full_source_from_model(self, m, nix, kminoverride=None):
        """Use the data from a model at a timestep nix to calculate the full source term S."""
        
        (alpha, beta), Cs, eta = self.get_vars_from_model(m, nix)
        C1, C2, C3, C4, C5, C6, C7 = Cs
        
        #Get component integrals
        J_A = self.J_A(self.k, alpha, C1, C2, eta, kminoverride)
        J_B = self.J_B(self.k, alpha, C3, C4, eta, kminoverride)
        J_C = self.J_C(self.k, alpha, beta, C5, eta, kminoverride)
        J_D = self.J_D(self.k, alpha, beta, C6, C7, eta, kminoverride)
        
        src = 1 / ((2*np.pi)**2) * (J_A + J_B + J_C + J_D)
        return src