'''generalsolution.py
Holds the general solution base class

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.

'''

import numpy as np

class GeneralSolution(object):
    """General solution base class."""
    
    def __init__(self, fixture, srcclass):
        """Create a GeneralSolution object."""
        self.srceqns = srcclass(fixture)
            
    def full_source_from_model(self, m, nix):
        pass
    
