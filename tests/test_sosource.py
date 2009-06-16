"""Test suite for sosource.py

Uses nosetest suite in numpy.testing
"""
import numpy as np
from numpy.testing import *
import sosource


def test_klessq():
    """Test klessq function"""
    #Setup, move to another func if necessary
    kmin = 0.001
    fullkmax = 1.026
    deltak = 0.001
    numsoks = 513
    nthetas = 129
    #Init vars
    fullk = np.arange(kmin, fullkmax+deltak, deltak)
    k = q = fullk[:numsoks]
    k2 = k[..., np.newaxis]
    q2 = q[np.newaxis, ...]
    theta = np.linspace(0,np.pi, nthetas)
    
    #Get calculated array
    klq = np.array([sosource.klessq(onek, q, theta) for onek in k])
    
    #Test cases
    assert_array_almost_equal(klq[:,:,-1], (k2 + q2))
    assert_array_almost_equal(klq[:,:,0], abs(k2 - q2))    

def check_conv():
    """Check convolution using analytic solution. Not a test function."""
    #Setup, move to another func if necessary
    kmin = 0.001
    fullkmax = 1.026
    deltak = 0.001
    numsoks = 513
    nthetas = 129
    A=1
    B=1
    #Init vars
    fullk = np.arange(kmin, fullkmax+deltak, deltak)
    k = q = fullk[:numsoks]
    k2 = k[..., np.newaxis]
    q2 = q[np.newaxis, ...]
    theta = np.linspace(0,np.pi, nthetas)
    ie = k, q, theta
    
    dp1=A/sqrt(fullk)
    dp1dot=-A/sqrt(fullk) -(A/B)*sqrt(fullk)*1j
    
    #Get calculated array
    klq = np.array([sosource.klessq(onek, q, theta) for onek in k])
    tterms=sosource.getthetaterms(ie, dp1, dp1dot)
    
    ansoln= (2*A/(3*k2*q2)) * ((k2+q2)**1.5-(abs(k2-q2))**1.5)
    aterm=tterms[0,0]+tterms[0,1]*1j
    env=array([max(abs((ansoln[:,i]-aterm[:,i])/ansoln[:,i])) for i in arange(len(ansoln))])
    return env
    
if __name__ == "__main__":
    run_module_suite()
