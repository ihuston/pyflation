""" test_canonical - test canonical models
"""
from numpy.testing import assert_, assert_raises, \
                          assert_equal,\
                          assert_array_almost_equal, \
                          assert_almost_equal, dec

import numpy as np

from pyflation import cosmomodels as c


class TestBg():

    def setup(self):
        self.fx =  {"potential_func": "msqphisq",
               "pot_params": {"nfields": 1},
               "nfields": 1,
               "ystart": np.array([12.0, 1/300.0,0]),
               "cq": 50,
               "solver": "rkdriver_tsix"}
        self.m = c.CanonicalBackground(**self.fx)

    @dec.slow
    def test_bgrun(self):
        self.m.run(saveresults=False)
        
class TestFO():
    
    def setup(self):
        self.fx = {"potential_func": "msqphisq",
                   "pot_params": {"nfields": 1},
                   "nfields": 1,
                   "ystart": np.array([18.0, -0.1,0,0,0]),
                   "k": np.array([1e-60]),
                   "cq": 50,
                   "solver": "rkdriver_tsix"}
        self.m = c.CanonicalFirstOrder(**self.fx)
        
    @dec.slow
    def test_fo(self):
        self.m.run(saveresults=False)
        
class TestTwoStage():
    
    def setup(self):
        self.fx = {"potential_func": "msqphisq",
                   "pot_params": {"nfields": 1},
                   "nfields": 1,
                   "bgystart": np.array([18.0, -0.1,0]),
                   "k": np.array([1e-60]),
                   "cq": 50,
                   "solver": "rkdriver_tsix"}
        self.m = c.FOCanonicalTwoStage(**self.fx)
        
    @dec.slow
    def test_two_stage(self):
        self.m.run(saveresults=False)