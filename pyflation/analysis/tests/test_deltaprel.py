''' test_deltaprel - Test functions for deltaprel module

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.
'''
import numpy as np
from numpy.testing import assert_, assert_raises, \
                          assert_array_almost_equal, \
                          assert_almost_equal
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
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([5,10]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        arr = deltaprel.soundspeeds(Vphi, phidot, H)
        actual = np.array([19/9.0, 92/72.0]).reshape((2,1,1))
        assert_array_almost_equal(arr, actual)
        
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, deltaprel.soundspeeds, self.Vphi, self.phidot, self.H)
        
class TestRhoDots():
    
    def setup(self):
        self.phidot = np.arange(24).reshape((4,3,2))
        self.H = np.arange(8).reshape((4,1,2))
    
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = deltaprel.rhodots(self.phidot, self.H)
        assert_(arr.shape == self.phidot.shape)
        
    def test_scalar(self):
        """Test results of 1x1x1 calculation."""
        arr = deltaprel.rhodots(1.7, 0.5)
        assert_almost_equal(arr, -3*0.5**3*1.7**2)
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        arr = deltaprel.rhodots(phidot, H)
        actual = np.array([-27, -864]).reshape((2,1,1))
        assert_array_almost_equal(arr, actual)
        
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, deltaprel.rhodots, self.phidot, self.H)