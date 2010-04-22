'''analyticsolution.py
Analytic solutions for the second order Klein-Gordon equation
Created on 22 Apr 2010

@author: Ian Huston
'''
from __future__ import division
import numpy as np
import scipy

#Change to fortran names for compatability 
Log = scipy.log 
Sqrt = scipy.sqrt
ArcTan = scipy.arctan
Pi = scipy.pi


class AnalyticSolution(object):
    """Analytic Solution base class.
    """
    
    def __init__(self, fixture, model):
        """Given a fixture and a cosmomodels model instance, initialises an AnalyticSolution class instance.
        """
        self.fx = fixture
        
        self.m = model
        
    
    def full_source_term(self):
        """Full source term after integration."""
        pass
    
class NoPhaseBunchDaviesSolution(AnalyticSolution):
    """Analytic solution using the Bunch Davies initial conditions as the first order 
    solution and with no phase information.
    
    \delta\varphi_1 = alpha/sqrt(k) 
    \dN{\delta\varphi_1} = -alpha/sqrt(k) - alpha/beta *sqrt(k)*1j 
    """
    
    def __init__(self, fixture, model):
        super(NoPhaseBunchDaviesSolution, self).__init__(fixture, model)
        
    
    def full_source_term(self):
        """Full source term after integration."""
        pass
    
    def J_A(self, k, alpha, C1, C2):
        """Solution for J_A which is the integral for A in terms of constants C1 and C2."""
        #Set limits from k
        kmin = k[0]
        kmax = k[-1]
        
        J_A = (-(alpha ** 2 * (Sqrt(2) * k * (2000 * C1 * k ** 2 + 951 * C2 * k ** 4) - (15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * Pi) / 2. - 
            15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * Log(2 * (Sqrt(k) + Sqrt(2) * Sqrt(k))))) / (2880. * k) - 
       (alpha ** 2 * (Sqrt(2) * k * (2000 * C1 * k ** 2 + 951 * C2 * k ** 4) - 
            15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * Log(2 * (Sqrt(k) + Sqrt(2) * Sqrt(k))) - 
            15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * Log(2 * Sqrt(k)))) / (2880. * k) + 
       (alpha ** 2 * (-(Sqrt(kmax) * Sqrt(-k + kmax) * 
               (80 * C1 * (3 * k ** 2 - 14 * k * kmax + 8 * kmax ** 2) + 
                 3 * C2 * (15 * k ** 4 + 10 * k ** 3 * kmax + 8 * k ** 2 * kmax ** 2 - 176 * k * kmax ** 3 + 128 * kmax ** 4))) + 
            Sqrt(kmax) * Sqrt(k + kmax) * (80 * C1 * (3 * k ** 2 + 14 * k * kmax + 8 * kmax ** 2) + 
               3 * C2 * (15 * k ** 4 - 10 * k ** 3 * kmax + 8 * k ** 2 * kmax ** 2 + 176 * k * kmax ** 3 + 128 * kmax ** 4)) - 
            15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * Log(2 * (Sqrt(kmax) + Sqrt(-k + kmax))) - 
            15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * Log(2 * (Sqrt(kmax) + Sqrt(k + kmax))))) / (2880. * k) + 
       (alpha ** 2 * (Sqrt(k - kmin) * Sqrt(kmin) * 
             (80 * C1 * (3 * k ** 2 - 14 * k * kmin + 8 * kmin ** 2) + 
               3 * C2 * (15 * k ** 4 + 10 * k ** 3 * kmin + 8 * k ** 2 * kmin ** 2 - 176 * k * kmin ** 3 + 128 * kmin ** 4)) + 
            Sqrt(kmin) * Sqrt(k + kmin) * (80 * C1 * (3 * k ** 2 + 14 * k * kmin + 8 * kmin ** 2) + 
               3 * C2 * (15 * k ** 4 - 10 * k ** 3 * kmin + 8 * k ** 2 * kmin ** 2 + 176 * k * kmin ** 3 + 128 * kmin ** 4)) - 
            15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * ArcTan(Sqrt(kmin) / Sqrt(k - kmin)) - 
            15 * k ** 3 * (16 * C1 + 3 * C2 * k ** 2) * Log(2 * (Sqrt(kmin) + Sqrt(k + kmin))))) / (2880. * k))
        
        return J_A
    
    def J_B(self, k, alpha, C3, C4):
        """Solution for J_B which is the integral for B in terms of constants C3 and C4."""
        kmax = k[-1]
        kmin = k[0]
        
        J_B = ((alpha ** 2 * (-(Sqrt(2) * k * (33488 * C3 * k ** 4 + 13833 * C4 * k ** 6)) + (105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * Pi) / 2. + 
            105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * Log(2 * (Sqrt(k) + Sqrt(2) * Sqrt(k))))) / (282240. * k ** 2) + 
       (alpha ** 2 * (-(Sqrt(2) * k * (33488 * C3 * k ** 4 + 13833 * C4 * k ** 6)) + 
            105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * Log(2 * (Sqrt(k) + Sqrt(2) * Sqrt(k))) + 
            105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * Log(2 * Sqrt(k)))) / (282240. * k ** 2) - 
       (alpha ** 2 * (-(Sqrt(kmax) * Sqrt(k + kmax) * 
               (112 * C3 * (105 * k ** 4 + 250 * k ** 3 * kmax - 104 * k ** 2 * kmax ** 2 - 48 * k * kmax ** 3 + 96 * kmax ** 4) + 
                 3 * C4 * (945 * k ** 6 - 630 * k ** 5 * kmax + 504 * k ** 4 * kmax ** 2 + 4688 * k ** 3 * kmax ** 3 - 
                    2176 * k ** 2 * kmax ** 4 - 1280 * k * kmax ** 5 + 2560 * kmax ** 6))) + 
            Sqrt(kmax) * Sqrt(-k + kmax) * (112 * C3 * 
                (105 * k ** 4 - 250 * k ** 3 * kmax - 104 * k ** 2 * kmax ** 2 + 48 * k * kmax ** 3 + 96 * kmax ** 4) + 
               3 * C4 * (945 * k ** 6 + 630 * k ** 5 * kmax + 504 * k ** 4 * kmax ** 2 - 4688 * k ** 3 * kmax ** 3 - 2176 * k ** 2 * kmax ** 4 + 
                  1280 * k * kmax ** 5 + 2560 * kmax ** 6)) + 
            105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * Log(2 * (Sqrt(kmax) + Sqrt(-k + kmax))) + 
            105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * Log(2 * (Sqrt(kmax) + Sqrt(k + kmax))))) / (282240. * k ** 2) - 
       (alpha ** 2 * (-(Sqrt(kmin) * Sqrt(k + kmin) * 
               (112 * C3 * (105 * k ** 4 + 250 * k ** 3 * kmin - 104 * k ** 2 * kmin ** 2 - 48 * k * kmin ** 3 + 96 * kmin ** 4) + 
                 3 * C4 * (945 * k ** 6 - 630 * k ** 5 * kmin + 504 * k ** 4 * kmin ** 2 + 4688 * k ** 3 * kmin ** 3 - 
                    2176 * k ** 2 * kmin ** 4 - 1280 * k * kmin ** 5 + 2560 * kmin ** 6))) - 
            Sqrt(k - kmin) * Sqrt(kmin) * (112 * C3 * 
                (105 * k ** 4 - 250 * k ** 3 * kmin - 104 * k ** 2 * kmin ** 2 + 48 * k * kmin ** 3 + 96 * kmin ** 4) + 
               3 * C4 * (945 * k ** 6 + 630 * k ** 5 * kmin + 504 * k ** 4 * kmin ** 2 - 4688 * k ** 3 * kmin ** 3 - 2176 * k ** 2 * kmin ** 4 + 
                  1280 * k * kmin ** 5 + 2560 * kmin ** 6)) + 
            105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * ArcTan(Sqrt(kmin) / Sqrt(k - kmin)) + 
            105 * k ** 5 * (112 * C3 + 27 * C4 * k ** 2) * Log(2 * (Sqrt(kmin) + Sqrt(k + kmin))))) / (282240. * k ** 2))
        return J_B
    