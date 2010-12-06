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
    
