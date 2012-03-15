""" test_reheating - tests for reheating module
"""
from numpy.testing import assert_, assert_raises, \
                          assert_equal,\
                          assert_array_almost_equal, \
                          assert_almost_equal, dec

import numpy as np

from pyflation import reheating
from pyflation import cosmomodels as c


class TestBgNoFluids():

    def setup(self):
        self.fx =  {"potential_func": "hybridquadratic",
               "pot_params": {"nfields": 2},
               "nfields": 2,
               "ystart": np.array([12.0, 1/300.0, 12.0,49/300.0,0,0,0]),
               "cq": 50,
               "solver": "rkdriver_tsix"}
        self.m = reheating.ReheatingBackground(**self.fx)

    @dec.slow
    def test_bgrun(self):
        self.m.run(saveresults=False)
        nofluid = c.CanonicalBackground(**self.fx)
        nofluid.run(saveresults=False)
        
        mend = self.m.findinflend()[1]
        nofluidend = nofluid.findinflend()[1]
        
        assert_array_almost_equal(self.m.yresult[:mend,0:5], nofluid.yresult[:nofluidend,0:5], 
                                  err_msg="Run without fluids does not equal standard background run.")

    def test_notransfers(self):
        assert_array_almost_equal(self.m.transfers, np.zeros_like(self.m.transfers),
                                  err_msg="Transfer coefficients should be zero.")
        
    def test_transfers_shape(self):
        newfx = self.fx.copy()
        newfx["transfers"] = np.zeros((newfx["nfields"],1))
        assert_raises(ValueError, reheating.ReheatingBackground, **newfx)