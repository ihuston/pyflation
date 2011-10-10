''' test_deltaprel - Test functions for deltaprel module

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.
'''
import numpy as np
from numpy.testing import assert_, assert_raises, \
                          assert_array_almost_equal, \
                          assert_almost_equal
from pyflation.analysis import deltaprel
from pyflation import cosmomodels as c

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
        
    def test_two_by_two_by_one(self):
        """Test results of 2x2x1 calculation."""
        Vphi = np.array([[1,2],[3,9]]).reshape((2,2,1))
        phidot = np.array([[5,1],[7,3]]).reshape((2,2,1))
        H = np.array([[2],[1]]).reshape((2,1,1))
        arr = deltaprel.soundspeeds(Vphi, phidot, H)
        actual = np.array([[31/30.0,4.0/3], [9.0/7,3]]).reshape((2,2,1))
        assert_array_almost_equal(arr, actual)

class TestPdots():
    
    def setup(self):
        self.Vphi = np.arange(1.0, 25.0).reshape((4,3,2))
        self.phidot = self.Vphi
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
    

    def test_shape(self):
        """Test whether the Pressures are shaped correctly."""    
        arr = deltaprel.Pdots(self.Vphi, self.phidot, self.H)
        assert_(arr.shape == self.Vphi.shape)
        
    def test_scalar(self):
        """Test results of 1x1x1 calculation."""
        arr = deltaprel.Pdots(3, 0.5, 2)
        assert_(arr == -6)
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([5,10]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        arr = deltaprel.Pdots(Vphi, phidot, H)
        actual = np.array([-57, -552]).reshape((2,1,1))
        assert_array_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test results of 2x2x1 calculation."""
        Vphi = np.array([[1,2],[3,9]]).reshape((2,2,1))
        phidot = np.array([[5,1],[7,3]]).reshape((2,2,1))
        H = np.array([[2],[1]]).reshape((2,1,1))
        arr = deltaprel.Pdots(Vphi, phidot, H)
        actual = np.array([[-310,-16], [-189,-81]]).reshape((2,2,1))
        assert_array_almost_equal(arr, actual)
        
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, deltaprel.Pdots, self.Vphi, self.phidot, self.H)

    def test_compare_cs(self):
        """Compare to result from cs^2 equations."""
        cs = deltaprel.soundspeeds(self.Vphi, self.phidot, self.H)
        rhodots = deltaprel.rhodots(self.phidot, self.H)
        prdots = deltaprel.Pdots(self.Vphi, self.phidot, self.H)
        assert_almost_equal(cs, prdots/rhodots)

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
        assert_almost_equal(arr, -3*0.5**2*1.7**2)
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        arr = deltaprel.rhodots(phidot, H)
        actual = np.array([-27, -432]).reshape((2,1,1))
        assert_array_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test results of 2x2x1 calculation."""
        Vphi = np.array([[1,2],[3,9]]).reshape((2,2,1))
        phidot = np.array([[5,1],[7,3]]).reshape((2,2,1))
        H = np.array([[2],[1]]).reshape((2,1,1))
        arr = deltaprel.rhodots(phidot, H)
        actual = np.array([[-300,-12], [-147,-27]]).reshape((2,2,1))
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
        assert_almost_equal(arr, -3*0.5**2*1.7**2)
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        arr = deltaprel.fullrhodot(phidot, H)
        actual = np.sum(np.array([-27, -432]).reshape((2,1,1)),axis=-1)
        assert_array_almost_equal(arr, actual)
        
    def test_wrongshape(self):
        """Test that wrong shapes raise exception."""
        self.H = np.arange(8).reshape((4,2))
        assert_raises(ValueError, deltaprel.fullrhodot, self.phidot, self.H)
        
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
        arr = deltaprel.deltarhosmatrix(self.Vphi, self.phidot, self.H, 
                                        self.modes, self.modesdot, self.axis)
        result = arr.shape
        actual = self.modes.shape
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
        
    def test_scalar(self):
        """Test results of scalar calculation with 1x1 mode matrix."""
        modes = np.array([[7]])
        modesdot = np.array([[3]])
        arr = deltaprel.deltarhosmatrix(3, 1.7, 0.5, modes, modesdot, axis=0)
        assert_almost_equal(arr, np.array([[0.5**2*1.7*3-0.5**4*1.7**2*1.7*7+21]]))
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([2,3]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        
        modes = np.array([10,5]).reshape((2,1,1,1))
        modesdot = np.array([10,5]).reshape((2,1,1,1))
        axis = 2
        
        arr = deltaprel.deltarhosmatrix(Vphi, phidot, H, modes, modesdot, axis)
        actual = np.array([-85, -4185]).reshape((2,1,1,1))
        assert_array_almost_equal(arr, actual)
        
    def test_extend_H(self):
        """Test that if H has no field axis it is created."""
        H = np.arange(8).reshape((4,2))
        arr = deltaprel.deltarhosmatrix(self.Vphi, self.phidot, H, #@UnusedVariable
                                        self.modes, self.modesdot, self.axis)
        #Test that no exception thrown about shape.
        
    def test_extend_Vphi(self):
        """Test that if Vphi has no k axis it is created."""
        Vphi = np.arange(12).reshape((4,3))
        arr = deltaprel.deltarhosmatrix(Vphi, self.phidot, self.H, #@UnusedVariable
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
        arr = deltaprel.deltarhosmatrix(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([[-4871,-12849],[-8024,-21194]]).reshape((2,2,1))
        assert_almost_equal(arr, desired)
                
    def test_std_result(self):
        """Test simple calculation with modes of shape (4,3,3,2)."""
        arr = deltaprel.deltarhosmatrix(self.Vphi, self.phidot, self.H, 
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


       [[[ -8.562000000000e+04,  -4.604285000000e+05],
         [ -9.247200000000e+04,  -4.960095000000e+05],
         [ -9.932400000000e+04,  -5.315905000000e+05]],

        [[ -1.523200000000e+05,  -7.612965000000e+05],
         [ -1.645280000000e+05,  -8.201655000000e+05],
         [ -1.767360000000e+05,  -8.790345000000e+05]],

        [[ -2.381000000000e+05,  -1.137416500000e+06],
         [ -2.572000000000e+05,  -1.225405500000e+06],
         [ -2.763000000000e+05,  -1.313394500000e+06]]],


       [[[ -8.229840000000e+06,  -2.067618150000e+07],
         [ -8.616504000000e+06,  -2.162613050000e+07],
         [ -9.003168000000e+06,  -2.257607950000e+07]],

        [[ -1.120210000000e+07,  -2.752791750000e+07],
         [ -1.172847200000e+07,  -2.879276250000e+07],
         [ -1.225484400000e+07,  -3.005760750000e+07]],

        [[ -1.463168000000e+07,  -3.535852950000e+07],
         [ -1.531926400000e+07,  -3.698327050000e+07],
         [ -1.600684800000e+07,  -3.860801150000e+07]]],


       [[[ -1.267685640000e+08,  -2.393502105000e+08],
         [ -1.309662720000e+08,  -2.471491595000e+08],
         [ -1.351639800000e+08,  -2.549481085000e+08]],

        [[ -1.565052000000e+08,  -2.923926705000e+08],
         [ -1.616877200000e+08,  -3.019201395000e+08],
         [ -1.668702400000e+08,  -3.114476085000e+08]],

        [[ -1.893720840000e+08,  -3.507394545000e+08],
         [ -1.956430960000e+08,  -3.621683155000e+08],
         [ -2.019141080000e+08,  -3.735971765000e+08]]]])
        

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
        arr = deltaprel.deltaPmatrix(self.Vphi, self.phidot, self.H, 
                                        self.modes, self.modesdot, self.axis)
        result = arr.shape
        actual = self.modes.shape
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
        
    def test_scalar(self):
        """Test results of scalar calculation with 1x1 mode matrix."""
        modes = np.array([[7]])
        modesdot = np.array([[3]])
        arr = deltaprel.deltaPmatrix(3, 1.7, 0.5, modes, modesdot, axis=0)
        assert_almost_equal(arr, np.array([[0.5**2*1.7*3-0.5**4*1.7**2*1.7*7-21]]))
        
    def test_two_by_one_by_one(self):
        """Test results of 2x1x1 calculation."""
        Vphi = np.array([2,3]).reshape((2,1,1))
        phidot = np.array([3,6]).reshape((2,1,1))
        H = np.array([1,2]).reshape((2,1,1))
        
        modes = np.array([10,5]).reshape((2,1,1,1))
        modesdot = np.array([10,5]).reshape((2,1,1,1))
        axis = 2
        
        arr = deltaprel.deltaPmatrix(Vphi, phidot, H, modes, modesdot, axis)
        actual = np.array([-125, -4215]).reshape((2,1,1,1))
        assert_array_almost_equal(arr, actual)
        
    def test_extend_H(self):
        """Test that if H has no field axis it is created."""
        H = np.arange(8).reshape((4,2))
        arr = deltaprel.deltaPmatrix(self.Vphi, self.phidot, H, #@UnusedVariable
                                        self.modes, self.modesdot, self.axis)
        #Test that no exception thrown about shape.
        
    def test_extend_Vphi(self):
        """Test that if Vphi has no k axis it is created."""
        Vphi = np.arange(12).reshape((4,3))
        arr = deltaprel.deltaPmatrix(Vphi, self.phidot, self.H, #@UnusedVariable
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
        arr = deltaprel.deltaPmatrix(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([[-4873,-12855],[-8032,-21214]]).reshape((2,2,1))
        assert_almost_equal(arr, desired)
                
    def test_std_result(self):
        """Test simple calculation with modes of shape (4,3,3,2)."""
        arr = deltaprel.deltaPmatrix(self.Vphi, self.phidot, self.H, 
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


       [[[ -8.583600000000e+04,  -4.606945000000e+05],
         [ -9.271200000000e+04,  -4.963035000000e+05],
         [ -9.958800000000e+04,  -5.319125000000e+05]],

        [[ -1.527040000000e+05,  -7.617465000000e+05],
         [ -1.649440000000e+05,  -8.206515000000e+05],
         [ -1.771840000000e+05,  -8.795565000000e+05]],

        [[ -2.387000000000e+05,  -1.138098500000e+06],
         [ -2.578400000000e+05,  -1.226131500000e+06],
         [ -2.769800000000e+05,  -1.314164500000e+06]]],


       [[[ -8.230704000000e+06,  -2.067714350000e+07],
         [ -8.617416000000e+06,  -2.162714450000e+07],
         [ -9.004128000000e+06,  -2.257714550000e+07]],

        [[ -1.120327600000e+07,  -2.752920750000e+07],
         [ -1.172970400000e+07,  -2.879411250000e+07],
         [ -1.225613200000e+07,  -3.005901750000e+07]],

        [[ -1.463321600000e+07,  -3.536019550000e+07],
         [ -1.532086400000e+07,  -3.698500450000e+07],
         [ -1.600851200000e+07,  -3.860981350000e+07]]],


       [[[ -1.267705080000e+08,  -2.393523005000e+08],
         [ -1.309682880000e+08,  -2.471513255000e+08],
         [ -1.351660680000e+08,  -2.549503505000e+08]],

        [[ -1.565076000000e+08,  -2.923952325000e+08],
         [ -1.616902000000e+08,  -3.019227855000e+08],
         [ -1.668728000000e+08,  -3.114503385000e+08]],

        [[ -1.893749880000e+08,  -3.507425365000e+08],
         [ -1.956460880000e+08,  -3.621714895000e+08],
         [ -2.019171880000e+08,  -3.736004425000e+08]]]])
    
    
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
        arr = deltaprel.deltaprelmodes(self.Vphi, self.phidot, self.H, 
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
        arr = deltaprel.deltaprelmodes(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, np.zeros_like(arr))
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([5.5,2.3]).reshape((2,1))
        phidot = np.array([2,5]).reshape((2,1))
        modes = np.array([[1/3.0,0.1],[0.1,0.5]]).reshape((2,2,1))
        modesdot = np.array([[0.1,0.2],[0.2,1/7.0]]).reshape((2,2,1))
        axis = 0
        H = np.array([3]).reshape((1,1))
        arr = deltaprel.deltaprelmodes(Vphi, phidot, H, modes, modesdot, axis)
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
        arr = deltaprel.deltaprelmodes(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([-2/3.0 -1j/3.0, 3 - 1j/3.0]).reshape((2,1))
        assert_almost_equal(arr, desired)
        
        
class TestDeltaPrelMatrixAlternate():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(1.0, 25.0).reshape((4,3,2))
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
    
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = deltaprel.deltaprelmodes_alternate(self.Vphi, self.phidot, self.H, 
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
        arr = deltaprel.deltaprelmodes_alternate(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, np.zeros_like(arr))
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([5.5,2.3]).reshape((2,1))
        phidot = np.array([2,5]).reshape((2,1))
        modes = np.array([[1/3.0,0.1],[0.1,0.5]]).reshape((2,2,1))
        modesdot = np.array([[0.1,0.2],[0.2,1/7.0]]).reshape((2,2,1))
        axis = 0
        H = np.array([3]).reshape((1,1))
        arr = deltaprel.deltaprelmodes_alternate(Vphi, phidot, H, modes, modesdot, axis)
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
        arr = deltaprel.deltaprelmodes_alternate(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([-2/3.0 -1j/3.0, 3 - 1j/3.0]).reshape((2,1))
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
        arr = deltaprel.deltaprelspectrum(self.Vphi, self.phidot, self.H, 
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
        arr = deltaprel.deltaprelspectrum(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, np.zeros_like(arr))
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([5.5,2.3]).reshape((2,1))
        phidot = np.array([2,5]).reshape((2,1))
        modes = np.array([[1/3.0,0.1],[0.1,0.5]]).reshape((2,2,1))
        modesdot = np.array([[0.1,0.2],[0.2,1/7.0]]).reshape((2,2,1))
        axis = 0
        H = np.array([3]).reshape((1,1))
        arr = deltaprel.deltaprelspectrum(Vphi, phidot, H, modes, modesdot, axis)
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
        arr = deltaprel.deltaprelspectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([9+2/3.0])
        assert_almost_equal(arr, desired)
        
class TestDeltaPrelSpectrumAlternate():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(1.0, 25.0).reshape((4,3,2))
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
        
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = deltaprel.deltaprelspectrum_alternate(self.Vphi, self.phidot, self.H, 
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
        arr = deltaprel.deltaprelspectrum_alternate(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, np.zeros_like(arr))
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([5.5,2.3]).reshape((2,1))
        phidot = np.array([2,5]).reshape((2,1))
        modes = np.array([[1/3.0,0.1],[0.1,0.5]]).reshape((2,2,1))
        modesdot = np.array([[0.1,0.2],[0.2,1/7.0]]).reshape((2,2,1))
        axis = 0
        H = np.array([3]).reshape((1,1))
        arr = deltaprel.deltaprelspectrum_alternate(Vphi, phidot, H, modes, modesdot, axis)
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
        arr = deltaprel.deltaprelspectrum_alternate(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([9+2/3.0])
        assert_almost_equal(arr, desired)


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
        arr = deltaprel.deltaPspectrum(self.Vphi, self.phidot, self.H, 
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
        actual = (0.5**2*1.7*3-0.5**4*1.7**2*1.7*7-21)**2
        arr = deltaprel.deltaPspectrum(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([7,9]).reshape((2,1))
        modes = np.array([[1,3],[2,5]]).reshape((2,2,1))
        modesdot = np.array([[1,3],[2,5]]).reshape((2,2,1))
        axis = 0
        H = np.array([2]).reshape((1,1))
        arr = deltaprel.deltaPspectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([12905**2 + 34069**2])
        assert_almost_equal(arr, desired)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = deltaprel.deltaPspectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([54])
        assert_almost_equal(arr, desired)
        

class TestComponentsFromModel():
    
    def setup(self):
        self.nfields = 2
        self.m = c.FOCanonicalTwoStage(nfields=2, potential_func="hybridquadratic",
                                       pot_params={"nfields":2},
                                       k=np.array([1e-62]))
        self.m.yresult = np.array([[[  1.32165292e+01 +0.00000000e+00j],
        [ -1.50754596e-01 +0.00000000e+00j],
        [  4.73189047e-13 +0.00000000e+00j],
        [ -3.63952781e-13 +0.00000000e+00j],
        [  5.40587341e-05 +0.00000000e+00j],
        [ -1.54862102e+89 -7.41671642e+88j],
        [ -2.13658450e+87 -1.10282252e+87j],
        [ -3.38643667e+88 -2.32461601e+88j],
        [  8.41541336e+68 +5.77708700e+68j],
        [ -2.56090533e+88 -1.39910039e+88j],
        [ -1.04340896e+76 -5.38567463e+75j],
        [  2.56090533e+88 +1.39910039e+88j],
        [ -9.29807779e+77 -5.05054386e+77j]]])
        
    def test_returned(self):
        """Test that the correct number of components are returned."""
        components = deltaprel.components_from_model(self.m)
        assert(len(components) == 6)
        
    def test_shapes(self):
        """Test that components returned are of correct shape."""
        components = deltaprel.components_from_model(self.m)
        shapes = [(1, self.nfields, 1), (1,self.nfields,1), (1, 1, 1), 
                  (1,self.nfields, self.nfields,1),
                  (1 ,self.nfields, self.nfields, 1), ()]
        for ix, var in enumerate(components):
            assert_(np.shape(var)==shapes[ix], msg="Shape of component %d is wrong" % ix)