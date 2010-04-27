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
        
        #Set ones array with same shape as self.k
        onekshape = np.ones(self.k.shape)
        
        #Set C_i values
        C1 = 1/H**2 * (Vppp + phidot/a**2 * (3 * a**2 * Vpp + 2 * self.k**2 ))
        
        C2 = 3.5 * phidot /((a*H)**2) * onekshape
        
        C3 = -4.5 / (a*H**2) * self.k
        
        C4 = -phidot/(a*H**2) / self.k
        
        C5 = -1.5 * phidot * onekshape
        
        C6 = 2 * phidot * self.k
        
        C7 = - phidot / self.k
        
        #Get component integrals
        J_A = self.J_A(self.k, alpha, C1, C2)
        J_B = self.J_B(self.k, alpha, C3, C4)
        J_C = self.J_C(self.k, alpha, beta, C5)
        J_D = self.J_D(self.k, alpha, beta, C6, C7)
        
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
        
    
    def J_A(self, k, alpha, C1, C2, eta):
        """Solution for J_A which is the integral for A in terms of constants C1 and C2."""
        #Set limits from k
        kmin = k[0]
        kmax = k[-1]
        
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