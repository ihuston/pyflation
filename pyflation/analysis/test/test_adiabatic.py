""" test_adiabatic - Test functions for adiabatic module

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


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
        
        self.m = self.getmodel(self.modes, k, lent, nfields, self.axis)

    def getmodel(self, modes, k, lent, nfields, axis):
        """Set up Cosmomodels instance for tests."""
        m = c.FOCanonicalTwoStage(k=k, nfields=nfields)
        m.yresult = np.zeros((lent, 2*nfields+1 + 2*nfields**2, len(k)), dtype=np.complex128)
        m.yresult[:,m.dps_ix] = utilities.flattenmodematrix(modes, nfields, ix1=axis, ix2=axis+1)
        return m
        
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = adiabatic.Pphi_modes(self.m)
        result = arr.shape
        actual = self.m.yresult[:,self.m.dps_ix].shape
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
    
    def test_singlefield(self):
        """Test single field calculation."""
        modes = np.array([[[7]]])
        axis=0
        k=np.array([5.25e-60])
        m=self.getmodel(modes, k, 1, 1, axis)
        actual = modes**2
        arr = adiabatic.Pphi_modes(m)
        assert_almost_equal(arr, actual)
        
    def test_one_by_two_by_two_by_one(self):
        """Test that 1x2x2x1 calculation works."""
        modes = np.array([[1,3],[2,5]]).reshape((1,2,2,1))
        axis = 1
        k = np.array([5.25e-60])
        m = self.getmodel(modes, k, 1, 2, axis)
        arr = adiabatic.Pphi_modes(m)
        desired = np.array([[10, 17],[17,29]]).reshape((1,4,1))
        assert_almost_equal(arr, desired)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((1,2,2,1))
        axis=1
        k = np.array([5.25e-60])
        m = self.getmodel(modes, k, 1, 2, axis)
        arr = adiabatic.Pphi_modes(m)
        desired = np.array([[2, -1+4*1j], [-1-4*1j, 11]]).reshape((1,4,1))
        assert_almost_equal(arr, desired)
        
    def test_off_diag_conjugates(self):
        """Test that off diagonal elements are conjugate."""
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((1,2,2,1))
        axis=1
        k = np.array([5.25e-60])
        m = self.getmodel(modes, k, 1, 2, axis)
        arr = adiabatic.Pphi_modes(m)
        assert_equal(arr[0,1], arr[0,2].conj())

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
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        axis=0
        arr = adiabatic.Pphi_matrix(modes, axis)
        assert_equal(arr[0,1], arr[1,0].conj())

class TestPr():
    
    def setup(self):
        self.axis = 1
        self.modes = np.arange(72.0).reshape((4.0,3,3,2))
        self.phidot = np.arange(24.0).reshape((4,3,2))
        lent = self.modes.shape[0]
        nfields = self.modes.shape[self.axis]
        k = np.array([1e-60, 5.25e-60])
        
        self.m = self.getmodel(self.phidot, self.modes, k, lent, nfields, self.axis)

    def getmodel(self, phidot, modes, k, lent, nfields, axis):
        """Set up Cosmomodels instance for tests."""
        m = c.FOCanonicalTwoStage(k=k, nfields=nfields)
        m.yresult = np.zeros((lent, 2*nfields+1 + 2*nfields**2, len(k)), dtype=np.complex128)
        m.yresult[:,m.dps_ix] = utilities.flattenmodematrix(modes, nfields, ix1=axis, ix2=axis+1)
        m.yresult[:,m.phidots_ix] = phidot
        return m
        
    def test_shape(self):
        """Test whether the rhodots are shaped correctly."""    
        arr = adiabatic.Pr(self.m)
        result = arr.shape
        newshape = list(self.m.yresult[:,self.m.dps_ix].shape)
        del newshape[self.axis]
        actual = tuple(newshape)
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
    
    def test_singlefield(self):
        """Test single field calculation."""
        modes = np.array([[[7]]])
        phidot = 1.7
        axis=0
        k=np.array([5.25e-60])
        m=self.getmodel(phidot, modes, k, 1, 1, axis)
        actual = np.array([[49/(1.7)**2]])
        arr = adiabatic.Pr(m)
        assert_almost_equal(arr, actual)
        
    def test_one_by_two_by_two_by_one(self):
        """Test that 1x2x2x1 calculation works."""
        modes = np.array([[1,3],[2,5]]).reshape((1,2,2,1))
        phidot = np.array([7,9]).reshape((1,2,1))
        axis = 1
        k = np.array([5.25e-60])
        m = self.getmodel(phidot, modes, k, 1, 2, axis)
        arr = adiabatic.Pr(m)
        desired = np.array([[4981/16900.0]]).reshape((1,1))
        assert_almost_equal(arr, desired)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((1,2,2,1))
        phidot = np.array([1,1]).reshape((1,2,1))
        axis=1
        k = np.array([5.25e-60])
        m = self.getmodel(phidot, modes, k, 1, 2, axis)
        arr = adiabatic.Pr(m)
        desired = np.array([[11/4.0]]).reshape((1,1))
        assert_almost_equal(arr, desired)
        
    def test_not_complex(self):
        """Test that result is not a complex object."""
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((1,2,2,1))
        phidot = np.array([1,1]).reshape((1,2,1))
        axis=1
        k = np.array([5.25e-60])
        m = self.getmodel(phidot, modes, k, 1, 2, axis)
        arr = adiabatic.Pr(m)
        assert_((not np.iscomplexobj(arr)))

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
        arr = adiabatic.Pr_spectrum(self.phidot, self.modes, self.axis)
        result = arr.shape
        newshape = list(self.Vphi.shape)
        del newshape[self.axis]
        actual = tuple(newshape)
        assert_(result == actual, "Result shape %s, but desired shape is %s"%(str(result), str(actual)))
    
    def test_singlefield(self):
        """Test single field calculation."""
        modes = np.array([[[7]]])
        phidot = 1.7
        axis=0
        actual = np.array([49/(1.7)**2])
        arr = adiabatic.Pr_spectrum(phidot, modes,axis)
        assert_almost_equal(arr, actual)
    
    def test_floatdivision(self):
        """Test that float division is used."""
        modes = np.array([[1,3],[2,5]]).reshape((1,2,2,1))
        phidot = np.array([7,9]).reshape((1,2,1))
        axis = 1
        arr = adiabatic.Pr_spectrum(phidot, modes, axis)
        desired = np.array([[4981.0/16900.0]]).reshape((1,1))
        assert_almost_equal(arr, desired)    
    
    def test_one_by_two_by_two_by_one(self):
        """Test that 1x2x2x1 calculation works."""
        modes = np.array([[1,3],[2,5]]).reshape((1,2,2,1))
        phidot = np.array([7,9]).reshape((1,2,1))
        axis = 1
        arr = adiabatic.Pr_spectrum(phidot, modes, axis)
        desired = np.array([[4981.0/16900.0]]).reshape((1,1))
        assert_almost_equal(arr, desired)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((1,2,2,1))
        phidot = np.array([1,1]).reshape((1,2,1))
        axis=1
        arr = adiabatic.Pr_spectrum(phidot, modes, axis)
        desired = np.array([[11/4.0]]).reshape((1,1))
        assert_almost_equal(arr, desired)
        
    def test_not_complex(self):
        """Test that result is not a complex object."""
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((1,2,2,1))
        phidot = np.array([1,1]).reshape((1,2,1))
        axis=1
        arr = adiabatic.Pr_spectrum(phidot, modes, axis)
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
        arr = adiabatic.Pzeta_spectrum(self.Vphi, self.phidot, self.H, 
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
        actual = (29.25)**2/9.0
        arr = adiabatic.Pzeta_spectrum(Vphi, phidot, H, modes, modesdot, axis)
        assert_almost_equal(arr, actual)
        
    def test_two_by_two_by_one(self):
        """Test that 2x2x1 calculation works."""
        Vphi = np.array([5.5,2.3]).reshape((2,1))
        phidot = np.array([2,5]).reshape((2,1))
        modes = np.array([[1/3.0,0.1],[0.1,0.5]]).reshape((2,2,1))
        modesdot = np.array([[0.1, 0.2],[0.2,1/7.0]]).reshape((2,2,1))
        axis = 0
        H = np.array([3]).reshape((1,1))
        arr = adiabatic.Pzeta_spectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([135451.600445493/(783**2)])
        assert_almost_equal(arr, desired)
        
    def test_imaginary(self):
        """Test calculation with complex values."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = adiabatic.Pzeta_spectrum(Vphi, phidot, H, modes, modesdot, axis)
        desired = np.array([38/36.0])
        assert_almost_equal(arr, desired)
        
    def test_not_complex(self):
        """Test that returned object is not complex."""
        Vphi = np.array([1,2]).reshape((2,1))
        phidot = np.array([1,1]).reshape((2,1))
        H = np.array([1]).reshape((1,1))
        modes = np.array([[1, 1j],[-1j, 3-1j]]).reshape((2,2,1))
        modesdot = np.array([[1, -1j],[1j, 3+1j]]).reshape((2,2,1))
        axis=0
        arr = adiabatic.Pzeta_spectrum(Vphi, phidot, H, modes, modesdot, axis)
        assert_((not np.iscomplexobj(arr)))
        