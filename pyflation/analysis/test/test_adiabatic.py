''' test_adiabatic - Test functions for adiabatic module

Author: Ian Huston
For license and copyright information see LICENSE.txt which was distributed with this file.
'''
import numpy as np
from numpy.testing import assert_, assert_raises, \
                          assert_equal,\
                          assert_array_almost_equal, \
                          assert_almost_equal
from pyflation.analysis import adiabatic, utilities
from pyflation import cosmomodels as c

class TestPphiModes():
    
    def setup(self):
        self.axis = 1
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        
        lent = self.modes.shape[0]
        nfields = self.modes.shape[self.axis]
        k = np.array([1e-60, 5.25e-60])
        
        m = self.getmodel(self.modes, k, lent, nfields)

    def getmodel(self, modes, k, lent, nfields):
        m = c.FOCanonicalTwoStage(k=k)
        m.yresult = np.zeros((lent, 2*nfields+1 + 2*nfields**2, len(k)), dtype=np.complex128)
        m.yresult[:,m.dps_ix] = modes[:]
        return m
        
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = adiabatic.Pphi_modes(self.modes, self.axis)
        result = arr.shape
        actual = self.modes.shape
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
    
    def test_singlefield(self):
        """Test single field calculation."""
        modes = np.array([[7]])
        axis=0
        actual = modes**2
        arr = adiabatic.Pphi_modes(modes, axis)
        assert_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        modes = np.array([[1,3],[2,5]]).reshape((2,2,1))
        axis = 0
        arr = adiabatic.Pphi_modes(modes, axis)
        desired = np.array([[10, 17],[17,29]]).reshape((2,2,1))
        assert_almost_equal(arr, desired)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        axis=0
        arr = adiabatic.Pphi_modes(modes, axis)
        desired = np.array([[2, -1+4*1j], [-1-4*1j, 11]]).reshape((2,2,1))
        assert_almost_equal(arr, desired)
        
    def test_off_diag_conjugates(self):
        """Test that off diagonal elements are conjugate."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = adiabatic.Pphi_modes(Vphi, phidot, H, modes, modesdot, axis)
        assert_equal(arr[0,1], arr[1,0].conj())

class TestPphiMatrix():
    
    def setup(self):
        self.axis=1
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        
        
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = adiabatic.Pphi_matrix(self.modes, self.axis)
        result = arr.shape
        actual = self.modes.shape
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
    
    def test_singlefield(self):
        """Test single field calculation."""
        modes = np.array([[7]])
        axis=0
        actual = modes**2
        arr = adiabatic.Pphi_matrix(modes, axis)
        assert_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        modes = np.array([[1,3],[2,5]]).reshape((2,2,1))
        axis = 0
        arr = adiabatic.Pphi_matrix(modes, axis)
        desired = np.array([[10, 17],[17,29]]).reshape((2,2,1))
        assert_almost_equal(arr, desired)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        axis=0
        arr = adiabatic.Pphi_matrix(modes, axis)
        desired = np.array([[2, -1+4*1j], [-1-4*1j, 11]]).reshape((2,2,1))
        assert_almost_equal(arr, desired)
        
    def test_off_diag_conjugates(self):
        """Test that off diagonal elements are conjugate."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = adiabatic.Pphi_matrix(Vphi, phidot, H, modes, modesdot, axis)
        assert_equal(arr[0,1], arr[1,0].conj())

class TestPrSpectrum():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(1.0, 25.0).reshape((4,3,2))
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
        
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = adiabatic.Pr(self.Vphi, self.phidot, self.H, 
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
        arr = adiabatic.Pr(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([7,9]).reshape((2,1))
        modes = np.array([[1,3],[2,5]]).reshape((2,2,1))
        modesdot = np.array([[1,3],[2,5]]).reshape((2,2,1))
        axis = 0
        H = np.array([2]).reshape((1,1))
        arr = adiabatic.Pr(Vphi, phidot, H, modes, modesdot, axis)
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
        arr = adiabatic.Pr(Vphi, phidot, H, modes, modesdot, axis)
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
        arr = adiabatic.Pr(Vphi, phidot, H, modes, modesdot, axis)
        assert_((not np.iscomplexobj(arr)))
        
class TestPzetaSpectrum():
    
    def setup(self):
        self.Vphi = np.arange(24.0).reshape((4,3,2))
        self.phidot = np.arange(1.0, 25.0).reshape((4,3,2))
        self.H = np.arange(1.0, 9.0).reshape((4,1,2))
        self.axis=1
        
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.modesdot = np.arange(10.0, 82.0).reshape((4.0,3,3,2))
        
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = adiabatic.Pzeta(self.Vphi, self.phidot, self.H, 
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
        arr = adiabatic.Pzeta(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([7,9]).reshape((2,1))
        modes = np.array([[1,3],[2,5]]).reshape((2,2,1))
        modesdot = np.array([[1,3],[2,5]]).reshape((2,2,1))
        axis = 0
        H = np.array([2]).reshape((1,1))
        arr = adiabatic.Pzeta(Vphi, phidot, H, modes, modesdot, axis)
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
        arr = adiabatic.Pzeta(Vphi, phidot, H, modes, modesdot, axis)
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
        arr = adiabatic.Pzeta(Vphi, phidot, H, modes, modesdot, axis)
        assert_((not np.iscomplexobj(arr)))
        