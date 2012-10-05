""" test_rk4 - tests for rk4.py

"""
#Author: Ian Huston
#For license and copyright information see LICENSE.txt which was distributed with this file.

import numpy as np
from numpy.testing import assert_, assert_raises, \
                          assert_equal,\
                          assert_array_almost_equal, \
                          assert_almost_equal

from pyflation import rk4
                          
class Test_rkdriver_tsix_basic():


    def setup(self):
        # Basic setup for rk4driver
        self.rkargs = dict(
                           ystart = np.zeros((1,1)),
                           simtstart = 0,
                           tsix = np.zeros((1,)),
                           tend = 10,
                           h = 0.01,
                           derivs = lambda y,x,k=None: x**2,
                           yarr = [],
                           xarr = [],
                           )
        self.x,self.y = rk4.rkdriver_tsix(**self.rkargs)

    def test_xresult(self):
        
        assert_almost_equal(self.x[-1], self.rkargs["tend"], 10, "End time not correct")
        
    def test_xarr_shape(self):
        number_steps = np.int(np.around((self.rkargs["tend"] - self.rkargs["simtstart"])
                                        /self.rkargs["h"]) + 1)
        assert_equal(self.x.shape, (number_steps,), "Result x array not correct shape")
        
    def test_yresult(self):
        assert_almost_equal(self.y[-1], 1e3/3.0, 10, "Final y result not correct")
        
    def test_yarr_shape(self):
        number_steps = np.int(np.around((self.rkargs["tend"] - self.rkargs["simtstart"])
                                        /self.rkargs["h"]) + 1)
        assert_equal(self.y.shape, (number_steps,1,1), "Result y array not correct shape")

class Test_rkdriver_tsix_2vars():


    def setup(self):
        # Basic setup for rk4driver
        self.rkargs = dict(
                           ystart = np.zeros((2,1)),
                           simtstart = 0,
                           tsix = np.zeros((1,)),
                           tend = 10,
                           h = 0.01,
                           derivs = lambda y,x,k=None: x**2,
                           yarr = [],
                           xarr = [],
                           )
        self.x,self.y = rk4.rkdriver_tsix(**self.rkargs)

    def test_xresult(self):
        
        assert_almost_equal(self.x[-1], self.rkargs["tend"], 10, "End time not correct")
        
    def test_xarr_shape(self):
        number_steps = np.int(np.around((self.rkargs["tend"] - self.rkargs["simtstart"])
                                        /self.rkargs["h"]) + 1)
        assert_equal(self.x.shape, (number_steps,), "Result x array not correct shape")
        
    def test_yresult(self):
        assert_almost_equal(self.y[-1], np.ones((2,1))*1e3/3.0, 10, "Final y result not correct")
        
    def test_yarr_shape(self):
        number_steps = np.int(np.around((self.rkargs["tend"] - self.rkargs["simtstart"])
                                        /self.rkargs["h"]) + 1)
        assert_equal(self.y.shape, (number_steps,2,1), "Result y array not correct shape")

class Test_rkdriver_tsix_difftsix():


    def setup(self):
        # Basic setup for rk4driver
        self.rkargs = dict(
                           ystart = np.zeros((1,2)),
                           simtstart = 0,
                           tsix = np.array([0, 500]),
                           tend = 10,
                           h = 0.01,
                           derivs = lambda y,x,k=None: x**2,
                           yarr = [],
                           xarr = [],
                           )
        self.x,self.y = rk4.rkdriver_tsix(**self.rkargs)

    def test_xresult(self):
        
        assert_almost_equal(self.x[-1], self.rkargs["tend"], 10, "End time not correct")
        
    def test_xarr_shape(self):
        number_steps = np.int(np.around((self.rkargs["tend"] - self.rkargs["simtstart"])
                                        /self.rkargs["h"]) + 1)
        assert_equal(self.x.shape, (number_steps,), "Result x array not correct shape")
        
    def test_yresult(self):
        desired = np.array([[1e3/3.0, 1e3/3.0 - 5**3/3.0 ]])
        assert_almost_equal(self.y[-1], desired, 10, "Final y result not correct")
        
    def test_yarr_shape(self):
        number_steps = np.int(np.around((self.rkargs["tend"] - self.rkargs["simtstart"])
                                        /self.rkargs["h"]) + 1)
        assert_equal(self.y.shape, (number_steps,1,2), "Result y array not correct shape")
        
        
        
###############################
# Rkdriver_append tests

class Test_rkdriver_append_basic():


    def setup(self):
        # Basic setup for rk4driver
        self.rkargs = dict(
                           ystart = np.zeros((1,1)),
                           simtstart = 0,
                           tsix = np.zeros((1,)),
                           tend = 10,
                           h = 0.01,
                           derivs = lambda y,x,k=None: x**2,
                           yarr=[],
                           xarr=[]
                           )
        self.x,self.y = rk4.rkdriver_append(**self.rkargs)
        self.x = np.hstack(self.x)
        self.y = np.vstack(self.y)

    def test_xresult(self):
        
        assert_almost_equal(self.x[-1], self.rkargs["tend"], 10, "End time not correct")
        
    def test_xarr_shape(self):
        number_steps = np.int(np.around((self.rkargs["tend"] - self.rkargs["simtstart"])
                                        /self.rkargs["h"]) + 1)
        assert_equal(self.x.shape, (number_steps,), "Result x array not correct shape")
        
    def test_yresult(self):
        assert_almost_equal(self.y[-1], 1e3/3.0, 10, "Final y result not correct")
        
    def test_yarr_shape(self):
        number_steps = np.int(np.around((self.rkargs["tend"] - self.rkargs["simtstart"])
                                        /self.rkargs["h"]) + 1)
        assert_equal(self.y.shape, (number_steps,1,1), "Result y array not correct shape")

class Test_rkdriver_append_2vars():


    def setup(self):
        # Basic setup for rk4driver
        self.rkargs = dict(
                           ystart = np.zeros((2,1)),
                           simtstart = 0,
                           tsix = np.zeros((1,)),
                           tend = 10,
                           h = 0.01,
                           derivs = lambda y,x,k=None: x**2,
                           yarr = [],
                           xarr = []
                           )
        self.x,self.y = rk4.rkdriver_append(**self.rkargs)
        self.x = np.hstack(self.x)
        self.y = np.vstack(self.y)

    def test_xresult(self):
        
        assert_almost_equal(self.x[-1], self.rkargs["tend"], 10, "End time not correct")
        
    def test_xarr_shape(self):
        number_steps = np.int(np.around((self.rkargs["tend"] - self.rkargs["simtstart"])
                                        /self.rkargs["h"]) + 1)
        assert_equal(self.x.shape, (number_steps,), "Result x array not correct shape")
        
    def test_yresult(self):
        assert_almost_equal(self.y[-1], np.ones((2,1))*1e3/3.0, 10, "Final y result not correct")
        
    def test_yarr_shape(self):
        number_steps = np.int(np.around((self.rkargs["tend"] - self.rkargs["simtstart"])
                                        /self.rkargs["h"]) + 1)
        assert_equal(self.y.shape, (number_steps,2,1), "Result y array not correct shape")

class Test_rkdriver_append_difftsix():


    def setup(self):
        # Basic setup for rk4driver
        self.rkargs = dict(
                           ystart = np.zeros((1,2)),
                           simtstart = 0,
                           tsix = np.array([0, 500]),
                           tend = 10,
                           h = 0.01,
                           derivs = lambda y,x,k=None: x**2,
                           yarr = [],
                           xarr = []
                           )
        self.x,self.y = rk4.rkdriver_append(**self.rkargs)
        self.x = np.hstack(self.x)
        self.y = np.vstack(self.y)

    def test_xresult(self):
        
        assert_almost_equal(self.x[-1], self.rkargs["tend"], 10, "End time not correct")
        
    def test_xarr_shape(self):
        number_steps = np.int(np.around((self.rkargs["tend"] - self.rkargs["simtstart"])
                                        /self.rkargs["h"]) + 1)
        assert_equal(self.x.shape, (number_steps,), "Result x array not correct shape")
        
    def test_yresult(self):
        desired = np.array([[1e3/3.0, 1e3/3.0 - 5**3/3.0 ]])
        assert_almost_equal(self.y[-1], desired, 10, "Final y result not correct")
        
    def test_yarr_shape(self):
        number_steps = np.int(np.around((self.rkargs["tend"] - self.rkargs["simtstart"])
                                        /self.rkargs["h"]) + 1)
        assert_equal(self.y.shape, (number_steps,1,2), "Result y array not correct shape")
        
####################################
# tests for rkdriver_rkf45
        
class Test_rkdriver_rkf45_basic():


    def setup(self):
        # Basic setup for rk4driver
        self.rkargs = dict(
                           ystart = np.zeros((1,1)),
                           xstart = 0,
                           xend = 10,
                           h = 0.01,
                           derivs = lambda y,x,k=None: x**2,
                           yarr=[],
                           xarr=[],
                           hmax=1,
                           hmin=1e-10,
                           abstol=0,
                           reltol=1e-6
                           )
        self.x,self.y = rk4.rkdriver_rkf45(**self.rkargs)
        self.x = np.hstack(self.x)
        self.y = np.vstack(self.y)

    def test_xresult(self):
        
        assert_almost_equal(self.x[-1], self.rkargs["xend"], 10, "End time not correct")
        
    def test_yresult(self):
        assert_almost_equal(self.y[-1], 1e3/3.0, 5, "Final y result not correct")
        
    def test_yarr_shape(self):
        number_steps = self.x.shape[0]
        assert_equal(self.y.shape, (number_steps,1,1), "Result y array not correct shape")

class Test_rkdriver_rkf45_2vars():


    def setup(self):
        # Basic setup for rk4driver
        self.rkargs = dict(
                           ystart = np.zeros((2,1)),
                           xstart = 0,
                           xend = 10,
                           h = 0.01,
                           derivs = lambda y,x,k=None: x**2,
                           yarr = [],
                           xarr = [],
                           hmax=1,
                           hmin=1e-10,
                           abstol=0,
                           reltol=1e-6
                           )
        self.x,self.y = rk4.rkdriver_rkf45(**self.rkargs)
        self.x = np.hstack(self.x)
        self.y = np.vstack(self.y)

    def test_xresult(self):
        
        assert_almost_equal(self.x[-1], self.rkargs["xend"], 10, "End time not correct")
        
    def test_yresult(self):
        assert_almost_equal(self.y[-1], np.ones((2,1))*1e3/3.0, 5, "Final y result not correct")
        
    def test_yarr_shape(self):
        number_steps = self.x.shape[0]
        assert_equal(self.y.shape, (number_steps,2,1), "Result y array not correct shape")

class Test_rkdriver_rkf45_difftsix():


    def setup(self):
        # Basic setup for rk4driver
        self.rkargs = dict(
                           ystart = np.zeros((1,2)),
                           xstart = np.array([0, 5]),
                           xend = 10,
                           h = 0.01,
                           derivs = lambda y,x,k=None: x**2,
                           yarr = [],
                           xarr = [],
                           hmax=1,
                           hmin=1e-10,
                           abstol=0,
                           reltol=1e-6
                           )
        self.x,self.y = rk4.rkdriver_rkf45(**self.rkargs)
        self.x = np.hstack(self.x)
        self.y = np.vstack(self.y)

    def test_xresult(self):
        
        assert_almost_equal(self.x[-1], self.rkargs["xend"], 10, "End time not correct")
       
    def test_yresult(self):
        desired = np.array([[1e3/3.0, 1e3/3.0 - 5**3/3.0 ]])
        assert_almost_equal(self.y[-1], desired, 5, "Final y result not correct")
        
    def test_yarr_shape(self):
        number_steps = self.x.shape[0]
        assert_equal(self.y.shape, (number_steps,1,2), "Result y array not correct shape")