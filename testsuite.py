"""Test suite to check cosmomodels.py"""
import unittest
import cosmomodels as c
import numpy as N
import numpy.testing as testing

class ModelTestCase(testing.NumpyTestCase):
    def testfirstplacenotspecial(self):
        k1 = 1e-61
        k2 = 1e-60
        mass = 2.95e-5
        self.twoks = c.TwoStageModel(k=N.array([k1,k2]), mass=mass)
        self.onek = c.TwoStageModel(k=N.array([k2]), mass=mass)        
        self.twoks.run()
        self.onek.run()
        
        twoy = self.twoks.yresult[:,:,1]
        oney = c.helpers.nanfillstart(self.onek.yresult[:,:,0], len(twoy))
        for varix in xrange(len(twoy[0])):
            testing.assert_almost_equal(twoy[:,varix], oney[:,varix], 3)
            
              

if __name__ == "__main__":
    unittest.main()