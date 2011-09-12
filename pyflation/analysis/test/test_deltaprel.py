''' test_deltaprel - Test functions for deltaprel module

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.
'''
import numpy as np
from numpy.testing import assert_, assert_raises, \
                          assert_array_almost_equal, \
                          assert_almost_equal
from pyflation.analysis import deltaprel

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
        
class TestFullRhoDot():
    
    def setup(self):
        self.phidot = np.arange(24).reshape((4,3,2))
        self.H = np.arange(8).reshape((4,1,2))
        self.axis=1
    
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = deltaprel.fullrhodot(self.phidot, self.H, self.axis)
        result = arr.shape
        newshape = list(self.phidot.shape)
        del newshape[self.axis]
        actual = tuple(newshape)
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
        
    def test_scalar(self):
        """Test results of 1x1x1 calculation."""
        arr = deltaprel.fullrhodot(1.7, 0.5)
        assert_almost_equal(arr, -3*0.5**3*1.7**2)
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        arr = deltaprel.fullrhodot(phidot, H)
        actual = np.sum(np.array([-27, -864]).reshape((2,1,1)),axis=-1)
        assert_array_almost_equal(arr, actual)
        
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, deltaprel.fullrhodot, self.phidot, self.H)
        
class TestDeltaRhosMatrix():
    
    def setup(self):
        self.Vphi = np.arange(24).reshape((4,3,2))
        self.phidot = np.arange(24).reshape((4,3,2))
        self.H = np.arange(8).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72).reshape((4,3,3,2))
    
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = deltaprel.deltarhosmatrix(self.Vphi, self.phidot, self.H, self.modes, self.axis)
        result = arr.shape
        actual = self.modes.shape
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
        
    def test_scalar(self):
        """Test results of scalar calculation with 1x1 mode matrix."""
        modes = np.array([[7]])
        arr = deltaprel.deltarhosmatrix(3, 1.7, 0.5, modes, axis=0)
        assert_almost_equal(arr, np.array([[0.5*1.7*7-0.5**4*1.7**2*1.7*7+21]]))
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([2,3]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        
        modes = np.array([10,5]).reshape((2,1,1,1))
        axis = 2
        
        arr = deltaprel.deltarhosmatrix(Vphi, phidot, H, modes, axis)
        actual = np.array([-85, -4245]).reshape((2,1,1,1))
        assert_array_almost_equal(arr, actual)
        
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, deltaprel.deltarhosmatrix, self.Vphi, self.phidot, 
                      self.H, self.modes, self.axis)
        
    def test_extend_Vphi(self):
        """Test that if Vphi has no k axis it is created."""
        Vphi = np.arange(12).reshape((4,3))
        arr = deltaprel.deltarhosmatrix(Vphi, self.phidot, self.H, #@UnusedVariable
                                        self.modes, self.axis)
        #Test that no exception thrown about shape.
        
        
    def test_std_result(self):
        """Test simple calculation with modes of shape (4,3,3,2)."""
        arr = deltaprel.deltarhosmatrix(self.Vphi, self.phidot, self.H, 
                                        self.modes, self.axis)
        assert_almost_equal(self.stdresult, arr, decimal=12)
           
    stdresult = np.array([[[[  0.0000000000000000e+00,  -4.1500000000000000e+01],
         [  0.0000000000000000e+00,  -4.6500000000000000e+01],
         [  0.0000000000000000e+00,  -5.1500000000000000e+01]],

        [[  1.2000000000000000e+01,  -3.4950000000000000e+02],
         [  1.6000000000000000e+01,  -4.1850000000000000e+02],
         [  2.0000000000000000e+01,  -4.8750000000000000e+02]],

        [[  4.8000000000000000e+01,  -9.5750000000000000e+02],
         [  5.6000000000000000e+01,  -1.1625000000000000e+03],
         [  6.4000000000000000e+01,  -1.3675000000000000e+03]]],


       [[[ -8.6076000000000000e+04,  -4.6185650000000000e+05],
         [ -9.2952000000000000e+04,  -4.9752150000000000e+05],
         [ -9.9828000000000000e+04,  -5.3318650000000000e+05]],

        [[ -1.5302400000000000e+05,  -7.6345650000000000e+05],
         [ -1.6526400000000000e+05,  -8.2243350000000000e+05],
         [ -1.7750400000000000e+05,  -8.8141050000000000e+05]],

        [[ -2.3910000000000000e+05,  -1.1404525000000000e+06],
         [ -2.5824000000000000e+05,  -1.2285735000000000e+06],
         [ -2.7738000000000000e+05,  -1.3166945000000000e+06]]],


       [[[ -8.2369440000000000e+06,  -2.0689051500000000e+07],
         [ -8.6238960000000000e+06,  -2.1639520500000000e+07],
         [ -9.0108480000000000e+06,  -2.2589989500000000e+07]],

        [[ -1.1211396000000000e+07,  -2.7544567500000000e+07],
         [ -1.1738104000000000e+07,  -2.8810012500000000e+07],
         [ -1.2264812000000000e+07,  -3.0075457500000000e+07]],

        [[ -1.4643456000000000e+07,  -3.5379439500000000e+07],
         [ -1.5331424000000000e+07,  -3.7004860500000000e+07],
         [ -1.6019392000000000e+07,  -3.8630281500000000e+07]]],


       [[[ -1.2680420400000000e+08,  -2.3940341050000000e+08],
         [ -1.3100299200000000e+08,  -2.4720395550000000e+08],
         [ -1.3520178000000000e+08,  -2.5500450050000000e+08]],

        [[ -1.5654840000000000e+08,  -2.9245676250000000e+08],
         [ -1.6173212000000000e+08,  -3.0198599550000000e+08],
         [ -1.6691584000000000e+08,  -3.1151522850000000e+08]],

        [[ -1.8942356400000000e+08,  -3.5081544650000000e+08],
         [ -1.9569589600000000e+08,  -3.6224623950000000e+08],
         [ -2.0196822800000000e+08,  -3.7367703250000000e+08]]]])
        
        
    
        
