""" test_nonadiabatic - Test functions for nonadiabatic module


"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


import numpy as np
from numpy.testing import assert_, assert_raises, \
                          assert_equal,\
                          assert_array_almost_equal, \
                          assert_almost_equal
from pyflation.analysis import nonadiabatic, utilities
from pyflation import cosmomodels as c


class TestSoundSpeeds():
    
    def setup(self):
        self.Vphi = np.arange(24).reshape((4,3,2))
        self.phidot = self.Vphi
        self.H = np.arange(8).reshape((4,1,2))
    

    def test_shape(self):
        """Test whether the soundspeeds are shaped correctly."""    
        arr = nonadiabatic.soundspeeds(self.Vphi, self.phidot, self.H)
        assert_(arr.shape == self.Vphi.shape)
        
    def test_scalar(self):
        """Test results of 1x1x1 calculation."""
        arr = nonadiabatic.soundspeeds(3, 0.5, 2)
        assert_(arr == 2)
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([5,10]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        arr = nonadiabatic.soundspeeds(Vphi, phidot, H)
        actual = np.array([19/9.0, 92/72.0]).reshape((2,1,1))
        assert_array_almost_equal(arr, actual)
        
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, nonadiabatic.soundspeeds, self.Vphi, self.phidot, self.H)
        
    def test_two_by_two_by_one(self):
        """Test results of 2x2x1 calculation."""
        Vphi = np.array([[1,2],[3,9]]).reshape((2,2,1))
        phidot = np.array([[5,1],[7,3]]).reshape((2,2,1))
        H = np.array([[2],[1]]).reshape((2,1,1))
        arr = nonadiabatic.soundspeeds(Vphi, phidot, H)
        actual = np.array([[31/30.0,4.0/3], [9.0/7,3]]).reshape((2,2,1))
        assert_array_almost_equal(arr, actual)
        
class TestTotalSoundSpeed():
    
    def setup(self):
        self.Vphi = np.arange(24).reshape((4,3,2))
        self.phidot = self.Vphi
        self.H = np.arange(8).reshape((4,1,2))
        self.axis = 1
    

    def test_shape(self):
        """Test whether the soundspeeds are shaped correctly."""    
        arr = nonadiabatic.totalsoundspeed(self.Vphi, self.phidot, self.H, self.axis)
        result = arr.shape
        newshape = list(self.phidot.shape)
        del newshape[self.axis]
        actual = tuple(newshape)
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
        
    def test_scalar(self):
        """Test results of 1x1x1 calculation."""
        arr = nonadiabatic.totalsoundspeed(3, 0.5, 2, 0)
        assert_equal(arr, 2)
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([5,10]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        arr = nonadiabatic.totalsoundspeed(Vphi, phidot, H, axis=1)
        actual = np.array([19/9.0, 92/72.0]).reshape((2,1))
        assert_array_almost_equal(arr, actual)
        
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, nonadiabatic.totalsoundspeed, self.Vphi, self.phidot, self.H, self.axis)
        
    def test_two_by_two_by_one(self):
        """Test results of 2x2x1 calculation."""
        Vphi = np.array([[1,2],[3,9]]).reshape((2,2,1))
        phidot = np.array([[5,1],[7,3]]).reshape((2,2,1))
        H = np.array([[2],[1]]).reshape((2,1,1))
        axis = 1
        arr = nonadiabatic.totalsoundspeed(Vphi, phidot, H, axis)
        actual = np.array([163/156.0,135.0/87.0]).reshape((2,1))
        assert_array_almost_equal(arr, actual)

class TestPdots():
    
    def setup(self):
        self.Vphi = np.arange(1.0, 25.0).reshape((4,3,2))
        self.phidot = self.Vphi
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
    

    def test_shape(self):
        """Test whether the Pressures are shaped correctly."""    
        arr = nonadiabatic.Pdots(self.Vphi, self.phidot, self.H)
        assert_(arr.shape == self.Vphi.shape)
        
    def test_scalar(self):
        """Test results of 1x1x1 calculation."""
        arr = nonadiabatic.Pdots(3, 0.5, 2)
        assert_(arr == -6)
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([5,10]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        arr = nonadiabatic.Pdots(Vphi, phidot, H)
        actual = np.array([-57, -552]).reshape((2,1,1))
        assert_array_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test results of 2x2x1 calculation."""
        Vphi = np.array([[1,2],[3,9]]).reshape((2,2,1))
        phidot = np.array([[5,1],[7,3]]).reshape((2,2,1))
        H = np.array([[2],[1]]).reshape((2,1,1))
        arr = nonadiabatic.Pdots(Vphi, phidot, H)
        actual = np.array([[-310,-16], [-189,-81]]).reshape((2,2,1))
        assert_array_almost_equal(arr, actual)
        
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, nonadiabatic.Pdots, self.Vphi, self.phidot, self.H)

    def test_compare_cs(self):
        """Compare to result from cs^2 equations."""
        cs = nonadiabatic.soundspeeds(self.Vphi, self.phidot, self.H)
        rhodots = nonadiabatic.rhodots(self.phidot, self.H)
        prdots = nonadiabatic.Pdots(self.Vphi, self.phidot, self.H)
        assert_almost_equal(cs, prdots/rhodots)

class TestFullPDot():
    
    def setup(self):
        self.Vphi = np.arange(1.0, 25.0).reshape((4,3,2))
        self.phidot = self.Vphi
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
        self.axis=1
    
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = nonadiabatic.fullPdot(self.Vphi, self.phidot, self.H, self.axis)
        result = arr.shape
        newshape = list(self.phidot.shape)
        del newshape[self.axis]
        actual = tuple(newshape)
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
        
    def test_scalar(self):
        """Test results of 1x1x1 calculation."""
        arr = nonadiabatic.fullPdot(3, 0.5, 2)
        assert_equal(arr, -6)
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([5,10]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        arr = nonadiabatic.fullPdot(Vphi, phidot, H)
        actual = np.sum(np.array([-57, -552]).reshape((2,1,1)),axis=-1)
        assert_array_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test results of 2x2x1 calculation."""
        Vphi = np.array([[1,2],[3,9]]).reshape((2,2,1))
        phidot = np.array([[5,1],[7,3]]).reshape((2,2,1))
        H = np.array([[2],[1]]).reshape((2,1,1))
        axis = 1
        arr = nonadiabatic.fullPdot(Vphi, phidot, H, axis)
        actual = np.array([[-310+-16], [-189+-81]]).reshape((2,1))
        assert_array_almost_equal(arr, actual)
        
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, nonadiabatic.fullPdot, self.Vphi, self.phidot, self.H)

class TestRhoDots():
    
    def setup(self):
        self.phidot = np.arange(24).reshape((4,3,2))
        self.H = np.arange(8).reshape((4,1,2))
    
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = nonadiabatic.rhodots(self.phidot, self.H)
        assert_(arr.shape == self.phidot.shape)
        
    def test_scalar(self):
        """Test results of 1x1x1 calculation."""
        arr = nonadiabatic.rhodots(1.7, 0.5)
        assert_almost_equal(arr, -3*0.5**2*1.7**2)
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        arr = nonadiabatic.rhodots(phidot, H)
        actual = np.array([-27, -432]).reshape((2,1,1))
        assert_array_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test results of 2x2x1 calculation."""
        Vphi = np.array([[1,2],[3,9]]).reshape((2,2,1))
        phidot = np.array([[5,1],[7,3]]).reshape((2,2,1))
        H = np.array([[2],[1]]).reshape((2,1,1))
        arr = nonadiabatic.rhodots(phidot, H)
        actual = np.array([[-300,-12], [-147,-27]]).reshape((2,2,1))
        assert_array_almost_equal(arr, actual)
        
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, nonadiabatic.rhodots, self.phidot, self.H)
        
class TestFullRhoDot():
    
    def setup(self):
        self.phidot = np.arange(24).reshape((4,3,2))
        self.H = np.arange(8).reshape((4,1,2))
        self.axis=1
    
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = nonadiabatic.fullrhodot(self.phidot, self.H, self.axis)
        result = arr.shape
        newshape = list(self.phidot.shape)
        del newshape[self.axis]
        actual = tuple(newshape)
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
        
    def test_scalar(self):
        """Test results of 1x1x1 calculation."""
        arr = nonadiabatic.fullrhodot(1.7, 0.5)
        assert_almost_equal(arr, -3*0.5**2*1.7**2)
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        arr = nonadiabatic.fullrhodot(phidot, H)
        actual = np.sum(np.array([-27, -432]).reshape((2,1,1)),axis=-1)
        assert_array_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test results of 2x2x1 calculation."""
        phidot = np.array([[5,1],[7,3]]).reshape((2,2,1))
        H = np.array([[2],[1]]).reshape((2,1,1))
        arr = nonadiabatic.fullrhodot(phidot, H, axis=1)
        actual = np.array([[-300+-12], [-147+-27]]).reshape((2,1))
        assert_array_almost_equal(arr, actual)
    
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, nonadiabatic.fullrhodot, self.phidot, self.H)
        
class TestDeltaRhosMatrix():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(24.0).reshape((4,3,2))
        self.H = np.arange(8.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
    
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = nonadiabatic.deltarhosmatrix(self.Vphi, self.phidot, self.H, 
                                        self.modes, self.modesdot, self.axis)
        result = arr.shape
        actual = self.modes.shape
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
        
    def test_scalar(self):
        """Test results of scalar calculation with 1x1 mode matrix."""
        modes = np.array([[7]])
        modesdot = np.array([[3]])
        arr = nonadiabatic.deltarhosmatrix(3, 1.7, 0.5, modes, modesdot, axis=0)
        assert_almost_equal(arr, np.array([[0.5**2*1.7*3-0.5**3*1.7**2*1.7*7+21]]))
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([2,3]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        
        modes = np.array([10,5]).reshape((2,1,1,1))
        modesdot = np.array([10,5]).reshape((2,1,1,1))
        axis = 2
        
        arr = nonadiabatic.deltarhosmatrix(Vphi, phidot, H, modes, modesdot, axis)
        actual = np.array([-85, -2025]).reshape((2,1,1,1))
        assert_array_almost_equal(arr, actual)
        
    def test_extend_H(self):
        """Test that if H has no field axis it is created."""
        H = np.arange(8).reshape((4,2))
        arr = nonadiabatic.deltarhosmatrix(self.Vphi, self.phidot, H, #@UnusedVariable
                                        self.modes, self.modesdot, self.axis)
        #Test that no exception thrown about shape.
        
    def test_extend_Vphi(self):
        """Test that if Vphi has no k axis it is created."""
        Vphi = np.arange(12).reshape((4,3))
        arr = nonadiabatic.deltarhosmatrix(Vphi, self.phidot, self.H, #@UnusedVariable
                                        self.modes, self.modesdot, self.axis)
        #Test that no exception thrown about shape.
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([7,9]).reshape((2,1))
        modes = np.array([[1,3],[2,5]]).reshape((2,2,1))
        modesdot = np.array([[1,3],[2,5]]).reshape((2,2,1))
        axis = 0
        H = np.array([2]).reshape((1,1))
        arr = nonadiabatic.deltarhosmatrix(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([[-2421,-6381],[-3974,-10502]]).reshape((2,2,1))
        assert_almost_equal(arr, desired)
                
    def test_std_result(self):
        """Test simple calculation with modes of shape (4,3,3,2)."""
        arr = nonadiabatic.deltarhosmatrix(self.Vphi, self.phidot, self.H, 
                                        self.modes, self.modesdot, self.axis)
        assert_almost_equal(arr, self.stdresult, decimal=12)
           
    stdresult = np.array([[[[  0.000000000000e+00,  -3.150000000000e+01],
         [  0.000000000000e+00,  -3.650000000000e+01],
         [  0.000000000000e+00,  -4.150000000000e+01]],

        [[  1.200000000000e+01,  -3.195000000000e+02],
         [  1.600000000000e+01,  -3.885000000000e+02],
         [  2.000000000000e+01,  -4.575000000000e+02]],

        [[  4.800000000000e+01,  -9.075000000000e+02],
         [  5.600000000000e+01,  -1.112500000000e+03],
         [  6.400000000000e+01,  -1.317500000000e+03]]],


       [[[ -4.242000000000e+04,  -1.521695000000e+05],
         [ -4.581600000000e+04,  -1.639365000000e+05],
         [ -4.921200000000e+04,  -1.757035000000e+05]],

        [[ -7.552000000000e+04,  -2.517255000000e+05],
         [ -8.158400000000e+04,  -2.712285000000e+05],
         [ -8.764800000000e+04,  -2.907315000000e+05]],

        [[ -1.181000000000e+05,  -3.762055000000e+05],
         [ -1.276000000000e+05,  -4.053885000000e+05],
         [ -1.371000000000e+05,  -4.345715000000e+05]]],


       [[[ -2.050512000000e+06,  -4.122631500000e+06],
         [ -2.146872000000e+06,  -4.312080500000e+06],
         [ -2.243232000000e+06,  -4.501529500000e+06]],

        [[ -2.791348000000e+06,  -5.489167500000e+06],
         [ -2.922584000000e+06,  -5.741512500000e+06],
         [ -3.053820000000e+06,  -5.993857500000e+06]],

        [[ -3.646208000000e+06,  -7.050979500000e+06],
         [ -3.817696000000e+06,  -7.375220500000e+06],
         [ -3.989184000000e+06,  -7.699461500000e+06]]],


       [[[ -2.109272400000e+07,  -3.414012150000e+07],
         [ -2.179123200000e+07,  -3.525262850000e+07],
         [ -2.248974000000e+07,  -3.636513550000e+07]],

        [[ -2.604120000000e+07,  -4.170666150000e+07],
         [ -2.690372000000e+07,  -4.306592850000e+07],
         [ -2.776624000000e+07,  -4.442519550000e+07]],

        [[ -3.151064400000e+07,  -5.002993350000e+07],
         [ -3.255445600000e+07,  -5.166065650000e+07],
         [ -3.359826800000e+07,  -5.329137950000e+07]]]])
        

class TestDeltaPMatrix():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(24.0).reshape((4,3,2))
        self.H = np.arange(8.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
    
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = nonadiabatic.deltaPmatrix(self.Vphi, self.phidot, self.H, 
                                        self.modes, self.modesdot, self.axis)
        result = arr.shape
        actual = self.modes.shape
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
        
    def test_scalar(self):
        """Test results of scalar calculation with 1x1 mode matrix."""
        modes = np.array([[7]])
        modesdot = np.array([[3]])
        arr = nonadiabatic.deltaPmatrix(3, 1.7, 0.5, modes, modesdot, axis=0)
        assert_almost_equal(arr, np.array([[0.5**2*1.7*3-0.5**3*1.7**2*1.7*7-21]]))
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([2,3]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        
        modes = np.array([10,5]).reshape((2,1,1,1))
        modesdot = np.array([10,5]).reshape((2,1,1,1))
        axis = 2
        
        arr = nonadiabatic.deltaPmatrix(Vphi, phidot, H, modes, modesdot, axis)
        actual = np.array([-125, -2055]).reshape((2,1,1,1))
        assert_array_almost_equal(arr, actual)
        
    def test_extend_H(self):
        """Test that if H has no field axis it is created."""
        H = np.arange(8).reshape((4,2))
        arr = nonadiabatic.deltaPmatrix(self.Vphi, self.phidot, H, #@UnusedVariable
                                        self.modes, self.modesdot, self.axis)
        #Test that no exception thrown about shape.
        
    def test_extend_Vphi(self):
        """Test that if Vphi has no k axis it is created."""
        Vphi = np.arange(12).reshape((4,3))
        arr = nonadiabatic.deltaPmatrix(Vphi, self.phidot, self.H, #@UnusedVariable
                                        self.modes, self.modesdot, self.axis)
        #Test that no exception thrown about shape.
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([7,9]).reshape((2,1))
        modes = np.array([[1,3],[2,5]]).reshape((2,2,1))
        modesdot = np.array([[1,3],[2,5]]).reshape((2,2,1))
        axis = 0
        H = np.array([2]).reshape((1,1))
        arr = nonadiabatic.deltaPmatrix(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([[-2423,-6387],[-3982,-10522]]).reshape((2,2,1))
        assert_almost_equal(arr, desired)
                
    def test_std_result(self):
        """Test simple calculation with modes of shape (4,3,3,2)."""
        arr = nonadiabatic.deltaPmatrix(self.Vphi, self.phidot, self.H, 
                                        self.modes, self.modesdot, self.axis)
        assert_almost_equal(arr, self.stdresult, decimal=12)
           
    stdresult = np.array([[[[  0.000000000000e+00,  -3.350000000000e+01],
         [  0.000000000000e+00,  -4.250000000000e+01],
         [  0.000000000000e+00,  -5.150000000000e+01]],

        [[ -1.200000000000e+01,  -3.615000000000e+02],
         [ -1.600000000000e+01,  -4.425000000000e+02],
         [ -2.000000000000e+01,  -5.235000000000e+02]],

        [[ -4.800000000000e+01,  -1.037500000000e+03],
         [ -5.600000000000e+01,  -1.262500000000e+03],
         [ -6.400000000000e+01,  -1.487500000000e+03]]],


       [[[ -4.263600000000e+04,  -1.524355000000e+05],
         [ -4.605600000000e+04,  -1.642305000000e+05],
         [ -4.947600000000e+04,  -1.760255000000e+05]],

        [[ -7.590400000000e+04,  -2.521755000000e+05],
         [ -8.200000000000e+04,  -2.717145000000e+05],
         [ -8.809600000000e+04,  -2.912535000000e+05]],

        [[ -1.187000000000e+05,  -3.768875000000e+05],
         [ -1.282400000000e+05,  -4.061145000000e+05],
         [ -1.377800000000e+05,  -4.353415000000e+05]]],


       [[[ -2.051376000000e+06,  -4.123593500000e+06],
         [ -2.147784000000e+06,  -4.313094500000e+06],
         [ -2.244192000000e+06,  -4.502595500000e+06]],

        [[ -2.792524000000e+06,  -5.490457500000e+06],
         [ -2.923816000000e+06,  -5.742862500000e+06],
         [ -3.055108000000e+06,  -5.995267500000e+06]],

        [[ -3.647744000000e+06,  -7.052645500000e+06],
         [ -3.819296000000e+06,  -7.376954500000e+06],
         [ -3.990848000000e+06,  -7.701263500000e+06]]],


       [[[ -2.109466800000e+07,  -3.414221150000e+07],
         [ -2.179324800000e+07,  -3.525479450000e+07],
         [ -2.249182800000e+07,  -3.636737750000e+07]],

        [[ -2.604360000000e+07,  -4.170922350000e+07],
         [ -2.690620000000e+07,  -4.306857450000e+07],
         [ -2.776880000000e+07,  -4.442792550000e+07]],

        [[ -3.151354800000e+07,  -5.003301550000e+07],
         [ -3.255744800000e+07,  -5.166383050000e+07],
         [ -3.360134800000e+07,  -5.329464550000e+07]]]])
    
    
class TestDeltaPrelMatrix():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(1.0, 25.0).reshape((4,3,2))
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
    
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = nonadiabatic.deltaPrelmodes(self.Vphi, self.phidot, self.H, 
                                       self.modes, self.modesdot, self.axis)
        result = arr.shape
        actual = self.Vphi.shape
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
    
    def test_singlefield(self):
        """Test single field calculation."""
        modes = np.array([[7]])
        modesdot = np.array([[3]])
        Vphi = 3
        phidot = 1.7
        H = 0.5
        axis=0
        arr = nonadiabatic.deltaPrelmodes(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, np.zeros_like(arr))
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([5.5,2.3]).reshape((2,1))
        phidot = np.array([2,5]).reshape((2,1))
        modes = np.array([[1/3.0,0.1],[0.1,0.5]]).reshape((2,2,1))
        modesdot = np.array([[0.1,0.2],[0.2,1/7.0]]).reshape((2,2,1))
        axis = 0
        H = np.array([3]).reshape((1,1))
        arr = nonadiabatic.deltaPrelmodes(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([0.31535513, 0.42954734370370623]).reshape((2,1))
        assert_almost_equal(arr, desired)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = nonadiabatic.deltaPrelmodes(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([-2/3.0 -1j/3.0, 3 - 1j/3.0]).reshape((2,1))
        assert_almost_equal(arr, desired)
        
        
class TestDeltaPnadMatrix():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(1.0, 25.0).reshape((4,3,2))
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
    
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = nonadiabatic.deltaPnadmodes(self.Vphi, self.phidot, self.H, 
                                       self.modes, self.modesdot, self.axis)
        result = arr.shape
        actual = self.Vphi.shape
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
    
    def test_singlefield(self):
        """Test single field calculation."""
        modes = np.array([[7]])
        modesdot = np.array([[5]])
        Vphi = 3
        phidot = 0.5
        H = 2
        axis=0
        arr = nonadiabatic.deltaPnadmodes(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, np.array([-71.25]))
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([2,3]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        
        modes = np.array([10,5]).reshape((2,1,1,1))
        modesdot = np.array([10,5]).reshape((2,1,1,1))
        axis = 2
        
        arr = nonadiabatic.deltaPnadmodes(Vphi, phidot, H, modes, modesdot, axis)
        actual = np.array([-2.22222222222, 138.75]).reshape((2,1,1))
        assert_array_almost_equal(arr, actual)
    
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([5.5,2.3]).reshape((2,1))
        phidot = np.array([2,5]).reshape((2,1))
        modes = np.array([[1/3.0,0.1],[0.1,0.5]]).reshape((2,2,1))
        modesdot = np.array([[0.1,0.2],[0.2,1/7.0]]).reshape((2,2,1))
        axis = 0
        H = np.array([3]).reshape((1,1))
        arr = nonadiabatic.deltaPnadmodes(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([3.884061, 16.1759427]).reshape((2,1))
        assert_almost_equal(arr, desired, decimal=5)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = nonadiabatic.deltaPnadmodes(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([-3.0 +4*1j, -18 + 3*1j]).reshape((2,1))
        assert_almost_equal(arr, desired)

class TestSModes():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(1.0, 25.0).reshape((4,3,2))
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
    
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = nonadiabatic.Smodes(self.Vphi, self.phidot, self.H, 
                                       self.modes, self.modesdot, self.axis)
        result = arr.shape
        actual = self.Vphi.shape
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
    
    def test_singlefield(self):
        """Test single field calculation."""
        modes = np.array([[7]])
        modesdot = np.array([[5]])
        Vphi = 3
        phidot = 0.5
        H = 2
        axis=0
        arr = nonadiabatic.Smodes(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, np.array([71.25/6.0]))
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([2,3]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        
        modes = np.array([10,5]).reshape((2,1,1,1))
        modesdot = np.array([10,5]).reshape((2,1,1,1))
        axis = 2
        
        arr = nonadiabatic.Smodes(Vphi, phidot, H, modes, modesdot, axis)
        actual = np.array([2.22222222222/39.0, -138.75/468.0]).reshape((2,1,1))
        assert_array_almost_equal(arr, actual)
    
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([5.5,2.3]).reshape((2,1))
        phidot = np.array([2,5]).reshape((2,1))
        modes = np.array([[1/3.0,0.1],[0.1,0.5]]).reshape((2,2,1))
        modesdot = np.array([[0.1,0.2],[0.2,1/7.0]]).reshape((2,2,1))
        axis = 0
        H = np.array([3]).reshape((1,1))
        arr = nonadiabatic.Smodes(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([3.884061/(-828.0), 16.1759427/(-828.0)]).reshape((2,1))
        assert_almost_equal(arr, desired, decimal=5)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = nonadiabatic.Smodes(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([0.25 -1/3.0*1j, +18/12.0 - 0.25*1j]).reshape((2,1))
        assert_almost_equal(arr, desired)
                
class TestDeltaPrelSpectrum():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(1.0, 25.0).reshape((4,3,2))
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
        
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = nonadiabatic.deltaPrelspectrum(self.Vphi, self.phidot, self.H, 
                                       self.modes, self.modesdot, self.axis)
        result = arr.shape
        newshape = list(self.phidot.shape)
        del newshape[self.axis]
        actual = tuple(newshape)
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
    
    def test_singlefield(self):
        """Test single field calculation."""
        modes = np.array([[7]])
        modesdot = np.array([[3]])
        Vphi = 3
        phidot = 1.7
        H = 0.5
        axis=0
        arr = nonadiabatic.deltaPrelspectrum(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, np.zeros_like(arr))
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([5.5,2.3]).reshape((2,1))
        phidot = np.array([2,5]).reshape((2,1))
        modes = np.array([[1/3.0,0.1],[0.1,0.5]]).reshape((2,2,1))
        modesdot = np.array([[0.1,0.2],[0.2,1/7.0]]).reshape((2,2,1))
        axis = 0
        H = np.array([3]).reshape((1,1))
        arr = nonadiabatic.deltaPrelspectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([0.31535513**2 + 0.42954734370370623**2])
        assert_almost_equal(arr, desired)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = nonadiabatic.deltaPrelspectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([9+2/3.0])
        assert_almost_equal(arr, desired)
        
    def test_not_complex(self):
        """Test that returned object is not complex."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = nonadiabatic.deltaPrelspectrum(Vphi, phidot, H, modes, modesdot, axis)
        assert_((not np.iscomplexobj(arr)))
        
class TestDeltaPnadSpectrum():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(1.0, 25.0).reshape((4,3,2))
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
        
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = nonadiabatic.deltaPnadspectrum(self.Vphi, self.phidot, self.H, 
                                       self.modes, self.modesdot, self.axis)
        result = arr.shape
        newshape = list(self.phidot.shape)
        del newshape[self.axis]
        actual = tuple(newshape)
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
    
    def test_singlefield(self):
        """Test single field calculation."""
        modes = np.array([[7]])
        modesdot = np.array([[5]])
        Vphi = 3
        phidot = 0.5
        H = 2
        axis=0
        arr = nonadiabatic.deltaPnadspectrum(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, (-71.25)**2)
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([5.5,2.3]).reshape((2,1))
        phidot = np.array([2,5]).reshape((2,1))
        modes = np.array([[1/3.0,0.1],[0.1,0.5]]).reshape((2,2,1))
        modesdot = np.array([[0.1,0.2],[0.2,1/7.0]]).reshape((2,2,1))
        axis = 0
        H = np.array([3]).reshape((1,1))
        arr = nonadiabatic.deltaPnadspectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([3.884061**2 + 16.1759427**2])
        assert_almost_equal(arr, desired, decimal=4)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = nonadiabatic.deltaPnadspectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([358])
        assert_almost_equal(arr, desired)
        
    def test_not_complex(self):
        """Test that returned object is not complex."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = nonadiabatic.deltaPnadspectrum(Vphi, phidot, H, modes, modesdot, axis)
        assert_((not np.iscomplexobj(arr)))


class TestDeltaPSpectrum():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(1.0, 25.0).reshape((4,3,2))
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
        
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = nonadiabatic.deltaPspectrum(self.Vphi, self.phidot, self.H, 
                                       self.modes, self.modesdot, self.axis)
        result = arr.shape
        newshape = list(self.phidot.shape)
        del newshape[self.axis]
        actual = tuple(newshape)
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
    
    def test_singlefield(self):
        """Test single field calculation."""
        modes = np.array([[7]])
        modesdot = np.array([[3]])
        Vphi = 3
        phidot = 1.7
        H = 0.5
        axis=0
        actual = (0.5**2*1.7*3-0.5**3*1.7**2*1.7*7-21)**2
        arr = nonadiabatic.deltaPspectrum(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([7,9]).reshape((2,1))
        modes = np.array([[1,3],[2,5]]).reshape((2,2,1))
        modesdot = np.array([[1,3],[2,5]]).reshape((2,2,1))
        axis = 0
        H = np.array([2]).reshape((1,1))
        arr = nonadiabatic.deltaPspectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([6405**2 + 16909**2])
        assert_almost_equal(arr, desired)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = nonadiabatic.deltaPspectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([54])
        assert_almost_equal(arr, desired)
        
    def test_not_complex(self):
        """Test that returned object is not complex."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = nonadiabatic.deltaPspectrum(Vphi, phidot, H, modes, modesdot, axis)
        assert_((not np.iscomplexobj(arr)))


class TestDeltaRhoSpectrum():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(1.0, 25.0).reshape((4,3,2))
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
        
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = nonadiabatic.deltarhospectrum(self.Vphi, self.phidot, self.H, 
                                       self.modes, self.modesdot, self.axis)
        result = arr.shape
        newshape = list(self.phidot.shape)
        del newshape[self.axis]
        actual = tuple(newshape)
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
    
    def test_singlefield(self):
        """Test single field calculation."""
        modes = np.array([[7]])
        modesdot = np.array([[5]])
        Vphi = 3
        phidot = 0.5
        H = 2
        axis=0
        arr = nonadiabatic.deltarhospectrum(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, (29.25)**2)
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([5.5,2.3]).reshape((2,1))
        phidot = np.array([2,5]).reshape((2,1))
        modes = np.array([[1/3.0,0.1],[0.1,0.5]]).reshape((2,2,1))
        modesdot = np.array([[0.1,0.2],[0.2,1/7.0]]).reshape((2,2,1))
        axis = 0
        H = np.array([3]).reshape((1,1))
        arr = nonadiabatic.deltarhospectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([139.38666666666**2+340.62142857**2])
        assert_almost_equal(arr, desired, decimal=5)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = nonadiabatic.deltarhospectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([38])
        assert_almost_equal(arr, desired)
        
    def test_not_complex(self):
        """Test that returned object is not complex."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = nonadiabatic.deltarhospectrum(Vphi, phidot, H, modes, modesdot, axis)
        assert_((not np.iscomplexobj(arr)))        


