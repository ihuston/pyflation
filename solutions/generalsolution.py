'''generalsolution.py
Holds the general solution base class

Created on 22 Apr 2010

@author: Ian Huston
'''

import numpy as np

class GeneralSolution(object):
    """General solution base class."""
    
    def __init__(self, fixture, srcclass):
        """Create a GeneralSolution object."""
        self.srceqns = srcclass(fixture)
            
    def full_source_from_model(self, m, nix):
        pass
    
    def J_A(self, C1, C2):
        pass
    
    def J_B(self, C3, C4):
        pass
    
    def J_C(self, C5):
        pass
    
    def J_D(self, C6, C7):
        pass
    
