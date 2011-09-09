''' test_deltaprel - Test functions for deltaprel module

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.
'''
import numpy as np
from numpy.testing import assert_, assert_raises

from pyflation.analysis import deltaprel
import nose




class TestSoundSpeeds():
    
    def setup(self):
        self.Vphi = np.arange(24).reshape((4,3,2))
        self.phidot = self.Vphi
        self.H = np.arange(8).reshape((4,1,2))
    

    def test_shape(self):
        """Test whether the soundspeeds are shaped correctly."""    
        arr = deltaprel.soundspeeds(self.Vphi, self.phidot, self.H)
        assert_(arr.shape == self.Vphi.shape)
        
    def test_scalar(self):
        """Test results of 1x1x1 calculation."""
        arr = deltaprel.soundspeeds(3, 0.5, 2)
        assert_(arr == 2)
        
        
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, deltaprel.soundspeeds, self.Vphi, self.phidot, self.H)
        