""" pyflation.analysis.test.test_utilities - Tests for utilities module


"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.


import numpy as np
from numpy.testing import assert_, assert_almost_equal, assert_equal

from pyflation.analysis import utilities
from pyflation import cosmomodels as c
from numpy.testing.utils import assert_raises



class TestComponentsFromModel():
    
    def setup(self):
        self.nfields = 2
        self.m = c.FOCanonicalTwoStage(nfields=2, potential_func="hybridquadratic",
                                       pot_params={"nfields":2},
                                       k=np.array([1e-62]))
        self.m.tresult = np.array([1.0])
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
        components = utilities.components_from_model(self.m)
        assert(len(components) == 6)
        
    def test_shapes(self):
        """Test that components returned are of correct shape."""
        components = utilities.components_from_model(self.m)
        shapes = [(1, self.nfields, 1), (1,self.nfields,1), (1, 1, 1), 
                  (1,self.nfields, self.nfields,1),
                  (1 ,self.nfields, self.nfields, 1), ()]
        for ix, var in enumerate(components):
            assert_(np.shape(var)==shapes[ix], msg="Shape of component %d is wrong" % ix)
            
    def test_negative_tix(self):
        """Test that negative tix is dealt with properly."""
        components = utilities.components_from_model(self.m, -1)
        shapes = [(1, self.nfields, 1), (1,self.nfields,1), (1, 1, 1), 
                  (1,self.nfields, self.nfields,1),
                  (1 ,self.nfields, self.nfields, 1), ()]
        for ix, var in enumerate(components):
            assert_(np.shape(var)==shapes[ix], msg="Shape of component %d is wrong" % ix)
            
class TestMakeSpectrum():
    
    def test_not_complex(self):
        """Test that return value is not a complex object."""
        modes_I = np.array([[1j, 1-3*1j]])
        arr = utilities.makespectrum(modes_I, axis=0)
        assert_(not np.iscomplexobj(arr))
        
    def test_scalar(self):
        """Test scalar value."""
        modes_I = 8
        arr = utilities.makespectrum(modes_I, axis=0)
        assert_almost_equal(arr, 64)
        
class TestSpectralIndex():
    
    def test_raise_on_zero(self):
        """Test that zero values cause ValueError"""
        y = np.array([0,0,0])
        x = np.array([1,2,3])
        assert_raises(ValueError, utilities.spectral_index, y, x, 0)
        
    def test_flat_line(self):
        """Test that flat lines give slope=1"""
        y = np.array([1,1,1])
        x = np.array([1,2,3])
        arr = utilities.spectral_index(y, x, 1)
        assert_equal(arr, 1)
    
    def test_45degree_line(self):
        """Test that upwards 45 degree line gives spectral index=2"""
        y = np.array([1,2,3])
        x = np.array([1,2,3])
        arr = utilities.spectral_index(y, x, 1)
        assert_almost_equal(arr, 2)   
        
    def test_neg45degree_line(self):
        """Test that downwards 45 degree line gives spectral index=2"""
        x = np.array([1.0,2.0,3.0])
        y = 1/x
        arr = utilities.spectral_index(y, x, 1)
        assert_almost_equal(arr, 0)   
         